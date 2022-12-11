use std::{
    alloc::{AllocError, Allocator, GlobalAlloc, Layout},
    cell::UnsafeCell,
    ptr::{self, NonNull},
    sync::Mutex,
};

use crate::{
    bucket::Bucket,
    realloc::{Realloc, ReallocMethod},
    AllocResult,
};

/// This is the main allocator, it contains multiple allocation buckets for
/// different sizes. Once you've read [`crate::header`], [`crate::block`],
/// [`crate::region`], [`crate::freelist`] and [`crate::bucket`], this is where
/// the circle gets completed:
///
/// ```text
///                                           Next Free Block                    Next Free Block
///                                |------------------------------------+   +-----------------------+
///                                |                                    |   |                       |
///               +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
///               |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// buckets[0] -> | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
///               |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///               +--------+------------------------+      ---------+-------------------------------------+
///
///                                           Next Free Block                    Next Free Block
///                                |------------------------------------+   +-----------------------+
///                                |                                    |   |                       |
///               +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
///               |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// buckets[1] -> | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
///               |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///               +--------+------------------------+      ---------+-------------------------------------+
///
/// ...........
///
///                                             Next Free Block                    Next Free Block
///                                  |------------------------------------+   +-----------------------+
///                                  |                                    |   |                       |
///                 +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
///                 |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// buckets[N-1] -> | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
///                 |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///                 +--------+------------------------+      ---------+-------------------------------------+
///
///                                           Next Free Block                    Next Free Block
///                                |------------------------------------+   +-----------------------+
///                                |                                    |   |                       |
///               +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
///               |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// dyn_bucket -> | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
///               |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///               +--------+------------------------+      ---------+-------------------------------------+
///
/// ```
///
/// Number of buckets and size of each bucket can be configured at compile
/// time. This struct is not thread safe and it also needs mutable borrows to
/// operate, so it has to be wrapped in [`UnsafeCell`] to satisfy
/// [`std::alloc::Allocator`] trait. See [`MmapAllocator`] for the public API.
///
/// # Drop
///
/// This struct doesn't implement [`Drop`] because region unmapping is
/// implemented by [`Bucket`]. If we don't implement [`Drop`], the compiler will
/// just call [`Drop::drop`] on all the struct members one by one, so all the
/// buckets will be dropped automatically.
struct InternalAllocator<const N: usize> {
    /// Size of each bucket, in bytes.
    sizes: [usize; N],
    /// Fixed size buckets.
    buckets: [Bucket; N],
    /// Any allocation request of `size > sizes[N - 1]` will use this bucket.
    dyn_bucket: Bucket,
}

impl<const N: usize> InternalAllocator<N> {
    /// Builds a new allocator configured with the given bucket sizes.
    pub const fn with_bucket_sizes(sizes: [usize; N]) -> Self {
        const BUCKET: Bucket = Bucket::new();
        InternalAllocator::<N> {
            sizes,
            buckets: [BUCKET; N],
            dyn_bucket: Bucket::new(),
        }
    }

    /// Returns the index of the [`Bucket`] where `layout` should be allocated.
    fn bucket_index_of(&self, layout: Layout) -> usize {
        for (i, size) in self.sizes.iter().enumerate() {
            if layout.size() <= *size {
                return i;
            }
        }

        self.buckets.len()
    }

    /// Returns a mutable reference to the [`Bucket`] at `index`.
    fn bucket_mut(&mut self, index: usize) -> &mut Bucket {
        if index == self.buckets.len() {
            &mut self.dyn_bucket
        } else {
            &mut self.buckets[index]
        }
    }

    /// Returns a mutable reference to the [`Bucket`] where `layout` should be
    /// allocated.
    fn dispatch(&mut self, layout: Layout) -> &mut Bucket {
        self.bucket_mut(self.bucket_index_of(layout))
    }

    /// Returns an address where `layout.size()` bytes can be safely written or
    /// [`AllocError`] if it fails to allocate.
    pub unsafe fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.dispatch(layout).allocate(layout)
    }

    /// Deallocates the memory block at `address`.
    pub unsafe fn deallocate(&mut self, address: NonNull<u8>, layout: Layout) {
        // We can find the bucket that has allocated the pointer because we also
        // know the layout. If the allocator trait changes and the layout is
        // no longer required, we can still obtain the block header given any
        // valid address and check the size to find the bucket. Let's hope it
        // doesn't change though, layouts are useful information for allocators!
        self.dispatch(layout).deallocate(address, layout)
    }

    /// Reallocation algorithm. Whether shrinking or growing, we'll try to
    /// preserve the maximum allocation size of each bucket as it was defined
    /// when creating the struct. So if `new_layout` should be allocated in a
    /// different bucket, we'll move the user contents there. Otherwise just
    /// delegate the call to the current bucket and handle reallocation
    /// internally.
    pub unsafe fn reallocate(&mut self, realloc: &Realloc) -> AllocResult {
        let current_bucket = self.bucket_index_of(realloc.old_layout);
        let ideal_bucket = self.bucket_index_of(realloc.new_layout);

        if current_bucket != ideal_bucket {
            let new_address = self.bucket_mut(ideal_bucket).allocate(realloc.new_layout)?;
            ptr::copy_nonoverlapping(
                realloc.address.as_ptr(),
                new_address.as_mut_ptr(),
                realloc.count(),
            );
            self.bucket_mut(current_bucket)
                .deallocate(realloc.address, realloc.old_layout);

            Ok(new_address)
        } else {
            self.bucket_mut(current_bucket).reallocate(realloc)
        }
    }
}

/// General purpose allocator. All memory is requested from the kernel using
/// [`libc::mmap`] and some tricks and optimizations are implemented such as
/// free list, block coalescing, block splitting and allocation buckets.
pub struct MmapAllocator<const N: usize = 3> {
    /// Currently we use a global [`Mutex`] to access the allocator, but here
    /// are some ideas to further optimize multithreaded allocations:
    ///
    /// 1. Use one [`Mutex`] per [`Bucket`]. That way different size allocations
    /// don't have to wait on each other. Note that reallocations might try to
    /// "move" a pointer from one [`Bucket`] to another if the requested new
    /// size changes drastically. If each [`Bucket`] has its own lock, we have
    /// to handle deadlocks properly with [`Mutex::try_lock`].
    ///
    /// 2. Don't use any [`Mutex`] at all, have one entire allocator per thread.
    /// Conceptually, we would need a mapping of [`std::thread::ThreadId`] to
    /// [`InternalAllocator`]. Instead of using general data structures that
    /// need to allocate memory, such as hash maps, we could use a fixed size
    /// array and store a tuple of `(ThreadId, Bucket)`. Each allocation will
    /// perform a linear scan to find the [`Bucket`] where we should allocate.
    /// This is technically O(n) but as long as we don't have thousands of
    /// threads it won't be an issue. If we end up needing to allocate
    /// memory for ourselves, we can just use [`crate::mmap`].
    allocator: Mutex<UnsafeCell<InternalAllocator<N>>>,
}

unsafe impl<const N: usize> Sync for MmapAllocator<N> {}

impl MmapAllocator {
    /// Default configuration includes 3 buckets of sizes 128, 1024 and 8192.
    pub const fn with_default_config() -> Self {
        Self {
            allocator: Mutex::new(UnsafeCell::new(InternalAllocator::with_bucket_sizes([
                128, 1024, 8192,
            ]))),
        }
    }
}

impl<const N: usize> MmapAllocator<N> {
    /// Builds a new allocator configured with the given bucket sizes.
    pub fn with_bucket_sizes(sizes: [usize; N]) -> Self {
        Self {
            allocator: Mutex::new(UnsafeCell::new(InternalAllocator::with_bucket_sizes(sizes))),
        }
    }
}

impl Default for MmapAllocator {
    fn default() -> Self {
        MmapAllocator::with_default_config()
    }
}

unsafe impl<const N: usize> Allocator for MmapAllocator<N> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            match self.allocator.lock() {
                Ok(mut allocator) => allocator.get_mut().allocate(layout),
                Err(_) => Err(AllocError),
            }
        }
    }

    unsafe fn deallocate(&self, address: NonNull<u8>, layout: Layout) {
        if let Ok(mut allocator) = self.allocator.lock() {
            allocator.get_mut().deallocate(address, layout)
        }
    }

    unsafe fn shrink(
        &self,
        address: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> AllocResult {
        match self.allocator.lock() {
            Ok(mut allocator) => allocator.get_mut().reallocate(&Realloc::new(
                address,
                old_layout,
                new_layout,
                ReallocMethod::Shrink,
            )),
            Err(_) => Err(AllocError),
        }
    }

    unsafe fn grow(
        &self,
        address: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> AllocResult {
        match self.allocator.lock() {
            Ok(mut allocator) => allocator.get_mut().reallocate(&Realloc::new(
                address,
                old_layout,
                new_layout,
                ReallocMethod::Grow,
            )),
            Err(_) => Err(AllocError),
        }
    }
}

unsafe impl GlobalAlloc for MmapAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match self.allocate(layout) {
            Ok(address) => address.cast().as_ptr(),
            Err(_) => ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocate(NonNull::new_unchecked(ptr), layout)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync,
        thread::{self, ThreadId},
    };

    use super::*;
    use crate::region::PAGE_SIZE;

    #[test]
    fn internal_allocator_wrapper() {
        let allocator = MmapAllocator::with_default_config();
        unsafe {
            let layout1 = Layout::array::<u8>(8).unwrap();
            let mut addr1 = allocator.allocate(layout1).unwrap();
            let slice1 = addr1.as_mut();

            slice1.fill(69);

            let layout2 = Layout::array::<u8>(PAGE_SIZE * 2).unwrap();
            let mut addr2 = allocator.allocate(layout2).unwrap();
            let slice2 = addr2.as_mut();

            slice2.fill(69);

            for value in slice1 {
                assert_eq!(value, &69);
            }

            allocator.deallocate(addr1.cast(), layout1);

            for value in slice2 {
                assert_eq!(value, &69);
            }

            allocator.deallocate(addr2.cast(), layout2);
        }
    }

    #[test]
    fn buckets() {
        unsafe {
            let sizes = [8, 16, 24];
            let mut allocator = InternalAllocator::<3>::with_bucket_sizes(sizes);

            macro_rules! verify_number_of_regions_per_bucket {
                ($expected:expr) => {
                    for i in 0..sizes.len() {
                        assert_eq!(allocator.buckets[i].regions().len(), $expected[i]);
                    }
                };
            }

            let layout1 = Layout::array::<u8>(sizes[0]).unwrap();
            let addr1 = allocator.allocate(layout1).unwrap().cast();
            verify_number_of_regions_per_bucket!([1, 0, 0]);

            let layout2 = Layout::array::<u8>(sizes[1]).unwrap();
            let addr2 = allocator.allocate(layout2).unwrap().cast();
            verify_number_of_regions_per_bucket!([1, 1, 0]);

            let layout3 = Layout::array::<u8>(sizes[2]).unwrap();
            let addr3 = allocator.allocate(layout3).unwrap().cast();
            verify_number_of_regions_per_bucket!([1, 1, 1]);

            allocator.deallocate(addr1, layout1);
            verify_number_of_regions_per_bucket!([0, 1, 1]);

            allocator.deallocate(addr2, layout2);
            verify_number_of_regions_per_bucket!([0, 0, 1]);

            allocator.deallocate(addr3, layout3);
            verify_number_of_regions_per_bucket!([0, 0, 0]);

            let layout4 = Layout::array::<u8>(sizes[2] + 128).unwrap();
            let addr4 = allocator.allocate(layout4).unwrap().cast();
            verify_number_of_regions_per_bucket!([0, 0, 0]);
            assert_eq!(allocator.dyn_bucket.regions().len(), 1);

            allocator.deallocate(addr4, layout4);
            assert_eq!(allocator.dyn_bucket.regions().len(), 0);
        }
    }

    fn verify_buckets_are_empty(allocator: MmapAllocator) {
        unsafe {
            let internal = allocator.allocator.lock().unwrap().get();
            for bucket in &(*internal).buckets {
                assert_eq!(bucket.regions().len(), 0);
            }
            assert_eq!((*internal).dyn_bucket.regions().len(), 0);
        }
    }

    /// We'll make all the threads do only allocs at the same time, then wait
    /// and do only deallocs at the same time.
    #[test]
    fn multiple_threads_synchronized_allocs_and_deallocs() {
        let allocator = MmapAllocator::with_default_config();

        let num_threads = 8;

        let barrier = sync::Barrier::new(num_threads);

        thread::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| unsafe {
                    let num_elements = 1024;
                    let layout = Layout::array::<ThreadId>(num_elements).unwrap();
                    let addr = allocator.allocate(layout).unwrap().cast::<ThreadId>();
                    let id = thread::current().id();

                    for i in 0..num_elements {
                        *addr.as_ptr().add(i) = id;
                    }

                    barrier.wait();

                    // Check memory corruption.
                    for i in 0..num_elements {
                        assert_eq!(*addr.as_ptr().add(i), id);
                    }

                    allocator.deallocate(addr.cast(), layout);
                });
            }
        });

        verify_buckets_are_empty(allocator);
    }

    /// In this case we'll make the threads do allocs and deallocs
    /// interchangeably.
    #[test]
    fn multiple_threads_unsynchronized_allocs_and_deallocs() {
        let allocator = MmapAllocator::with_default_config();

        let num_threads = 8;

        let barrier = sync::Barrier::new(num_threads);

        thread::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| unsafe {
                    // We'll use different sizes to make sure that contention
                    // over a single region or multiple regions doesn't cause
                    // issues.
                    let layouts = [16, 256, 1024, 2048, 4096, 8192]
                        .map(|size| Layout::array::<u8>(size).unwrap());

                    // Miri is really slow, but we don't need as many operations
                    // to find bugs with it.
                    let num_allocs = if cfg!(miri) { 20 } else { 2000 };

                    for layout in layouts {
                        barrier.wait();
                        for _ in 0..num_allocs {
                            let addr = allocator.allocate(layout).unwrap().cast::<u8>();
                            if cfg!(miri) {
                                // Since Miri is slow we won't write all the
                                // bytes, just a few to check data races. If
                                // somehow two threads receive the same address,
                                // Miri will catch that.
                                let offsets = [0, layout.size() / 2, layout.size() - 1];
                                let values = [1, 5, 10];
                                for (offset, value) in offsets.iter().zip(values) {
                                    *addr.as_ptr().add(*offset) = value;
                                }
                                for (offset, value) in offsets.iter().zip(values) {
                                    assert_eq!(*addr.as_ptr().add(*offset), value);
                                }
                            } else {
                                // If we're not using Miri then write all the
                                // bytes and check them again later.
                                for i in 0..layout.size() {
                                    *addr.as_ptr().add(i) = (i % 256) as u8;
                                }
                                for i in 0..layout.size() {
                                    assert_eq!(*addr.as_ptr().add(i), (i % 256) as u8);
                                }
                            }
                            allocator.deallocate(addr, layout);
                        }
                    }
                });
            }
        });

        verify_buckets_are_empty(allocator);
    }
}
