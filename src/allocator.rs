use std::{
    alloc::{AllocError, Allocator, GlobalAlloc, Layout},
    ptr::{self, NonNull},
    sync::Mutex,
};

use crate::{bucket::Bucket, realloc::Realloc, AllocResult};

/// This is the main allocator, it contains multiple allocation buckets for
/// different sizes. Once you've read [`crate::header`], [`crate::block`],
/// [`crate::region`], [`crate::freelist`] and [`crate::bucket`], this is where
/// the circle gets completed:
///
/// ```text
///                                           Next Free Block                    Next Free Block
///                                +------------------------------------+   +-----------------------+
///                                |                                    |   |                       |
///               +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
///               |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// buckets[0] -> | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
///               |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///               +--------+------------------------+      +--------+-------------------------------------+
///
///                                                         Next Free Block
///                                           +----------------------------------------+
///                                           |                                        |
///               +--------+------------------|-----+      +--------+------------------|------------------+
///               |        | +-------+    +---|---+ |      |        | +-------+    +---|---+    +-------+ |
/// buckets[1] -> | Region | | Block | -> | Free  | | ---> | Region | | Block | -> | Free  | -> | Block | |
///               |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///               +--------+------------------------+      +--------+-------------------------------------+
///
/// .......................................................................................................
///
///                                                     Next Free Block
///                                             +---------------------------+
///                                             |                           |
///                 +--------+------------------|-----+      +--------+-----|-------------------------------+
///                 |        | +-------+    +---|---+ |      |        | +---|---+    +-------+    +-------+ |
/// buckets[N-1] -> | Region | | Block | -> | Free  | | ---> | Region | | Free  | -> | Block | -> | Block | |
///                 |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///                 +--------+------------------------+      +--------+-------------------------------------+
///
///                                                     Next Free Block
///                                +-----------------------------------------------------+
///                                |                                                     |
///                 +--------+-----|------------------+      +--------+------------------|------------------+
///                 |        | +---|---+    +-------+ |      |        | +-------+    +---|---+    +-------+ |
/// dyn_bucket ->   | Region | | Free  | -> | Block | | ---> | Region | | Block | -> | Free  | -> | Block | |
///                 |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
///                 +--------+------------------------+      +--------+-------------------------------------+
/// ```
///
/// Number of buckets and size of each bucket can be configured at compile
/// time. This struct is not thread safe and it also needs mutable borrows to
/// operate, so it has to be wrapped in some container like [`Mutex`] to satisfy
/// [`std::alloc::Allocator`] trait. See [`Rulloc`] for the public API.
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
    #[inline]
    fn dispatch(&mut self, layout: Layout) -> &mut Bucket {
        self.bucket_mut(self.bucket_index_of(layout))
    }

    /// Returns an address where `layout.size()` bytes can be safely written or
    /// [`AllocError`] if it fails to allocate.
    #[inline]
    pub unsafe fn allocate(&mut self, layout: Layout) -> AllocResult {
        self.dispatch(layout).allocate(layout)
    }

    /// Deallocates the memory block at `address`.
    #[inline]
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

        if current_bucket == ideal_bucket {
            return self.bucket_mut(current_bucket).reallocate(realloc);
        }

        let new_address = self.bucket_mut(ideal_bucket).allocate(realloc.new_layout)?;
        ptr::copy_nonoverlapping(
            realloc.address.as_ptr(),
            new_address.as_mut_ptr(),
            realloc.count(),
        );
        self.bucket_mut(current_bucket)
            .deallocate(realloc.address, realloc.old_layout);

        Ok(new_address)
    }
}

/// This struct exposes the public interface by implementing
/// [`std::alloc::Allocator`].
///
/// # Examples
///
/// ## Standalone allocator
///
/// ```rust
/// #![feature(allocator_api)]
/// #![feature(slice_ptr_get)]
///
/// use std::alloc::{Allocator, Layout};
///
/// use rulloc::Rulloc;
///
/// let rulloc = Rulloc::default();
/// let (size, align) = (128, 8);
/// let layout = Layout::from_size_align(size, align).unwrap();
///
/// unsafe {
///     let address = rulloc.allocate(layout).unwrap();
///     // The allocator can return more space than requested.
///     assert!(address.len() >= size);
///     // Alignment is guaranteed for any power of two.
///     assert_eq!(address.as_mut_ptr() as usize % align, 0);
///     // Deallocate the pointer.
///     rulloc.deallocate(address.cast(), layout);
/// }
/// ```
///
/// ## Collections and [`Box`]
///
/// ```no_run
/// #![feature(allocator_api)]
///
/// use std::alloc::Allocator;
///
/// use rulloc::Rulloc;
///
/// let rulloc = Rulloc::default();
///
/// // Any struct that supports the allocator API works with Rulloc.
/// let mut num = Box::new_in(12, &rulloc);
/// assert_eq!(*num, 12);
///
/// let mut vec = Vec::new_in(&rulloc);
/// vec.push(5);
/// assert_eq!(vec[0], 5);
/// ```
///
/// ## Global allocator
///
/// ```no_run
/// #![feature(allocator_api)]
///
/// use rulloc::Rulloc;
///
/// #[global_allocator]
/// static ALLOCATOR: Rulloc = Rulloc::with_default_config();
///
/// fn main() {
///     let num = Box::new(5);
///     assert_eq!(*num, 5);
/// }
/// ```
pub struct Rulloc<const N: usize = 3> {
    /// Currently we use a global [`Mutex`] to access the allocator, but here
    /// are some ideas to further optimize multithreaded allocations:
    ///
    /// 1. Use one [`Mutex`] per [`Bucket`]. That way different size allocations
    /// don't have to wait on each other. Note that reallocations might try to
    /// "move" a pointer from one [`Bucket`] to another if the requested new
    /// size changes drastically. If each [`Bucket`] has its own lock, we have
    /// to handle deadlocks properly with [`Mutex::try_lock`].
    ///
    /// 2. Use a fixed number of allocators and distribute requests from
    /// different threads between them (round-robin, for example). Each
    /// allocator could have a global [`Mutex`] or one [`Mutex`] per [`Bucket`]
    /// like mentioned above.
    ///
    /// 3. Don't use any [`Mutex`] at all, have one entire allocator per thread.
    /// Conceptually, we would need a mapping of [`std::thread::ThreadId`] to
    /// [`InternalAllocator`]. Instead of using general data structures that
    /// need to allocate memory, such as hash maps, we could use a fixed size
    /// array and store a tuple of `(ThreadId, Bucket)`. Each allocation will
    /// perform a linear scan to find the [`Bucket`] where we should allocate.
    /// This is technically O(n) but as long as we don't have thousands of
    /// threads it won't be an issue. If we end up needing to allocate memory
    /// for ourselves, we can just use [`crate::platform::request_memory`]. The
    /// issue with this approach is that we have to deal with threads that
    /// deallocate memory which was not allocated by themselves, so we need more
    /// than a simple mapping.
    allocator: Mutex<InternalAllocator<N>>,
}

unsafe impl<const N: usize> Sync for Rulloc<N> {}

impl Rulloc {
    /// Default configuration includes 3 buckets of sizes 128, 1024 and 8192.
    /// See [`Rulloc::<N>::with_bucket_sizes`] for details.
    pub const fn with_default_config() -> Self {
        Self {
            allocator: Mutex::new(InternalAllocator::with_bucket_sizes([128, 1024, 8192])),
        }
    }
}

impl<const N: usize> Rulloc<N> {
    /// Builds a new allocator configured with the given bucket sizes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(allocator_api)]
    ///
    /// use std::alloc::{Allocator, Layout};
    ///
    /// use rulloc::Rulloc;
    ///
    /// // 3 fixed size buckets. First one will contain allocations less than
    /// // or equal to 64 bytes in size, second one will contain allocations
    /// // less than or equal to 128 bytes, and so forth. Allocations larger
    /// // than the last bucket size will be allocated separately.
    /// let rulloc = Rulloc::<3>::with_bucket_sizes([64, 128, 256]);
    ///
    /// // Allocated in the first bucket.
    /// let p1 = rulloc.allocate(Layout::from_size_align(64, 8).unwrap()).unwrap();
    /// // Allocated in the second bucket.
    /// let p2 = rulloc.allocate(Layout::from_size_align(100, 8).unwrap()).unwrap();
    /// // Allocated in the third bucket.
    /// let p3 = rulloc.allocate(Layout::from_size_align(210, 8).unwrap()).unwrap();
    /// // Allocated in a dynamic bucket that can allocate any size.
    /// let p4 = rulloc.allocate(Layout::from_size_align(512, 8).unwrap()).unwrap();
    ///
    /// assert!(p1.len() >= 64);
    /// assert!(p2.len() >= 100);
    /// assert!(p3.len() >= 210);
    /// assert!(p4.len() >= 512);
    /// ```
    pub fn with_bucket_sizes(sizes: [usize; N]) -> Self {
        Self {
            allocator: Mutex::new(InternalAllocator::with_bucket_sizes(sizes)),
        }
    }
}

impl Default for Rulloc {
    fn default() -> Self {
        Rulloc::with_default_config()
    }
}

unsafe impl<const N: usize> Allocator for Rulloc<N> {
    fn allocate(&self, layout: Layout) -> AllocResult {
        unsafe {
            match self.allocator.lock() {
                Ok(mut allocator) => allocator.allocate(layout),
                Err(_) => Err(AllocError),
            }
        }
    }

    unsafe fn deallocate(&self, address: NonNull<u8>, layout: Layout) {
        if let Ok(mut allocator) = self.allocator.lock() {
            allocator.deallocate(address, layout)
        }
    }

    unsafe fn shrink(
        &self,
        address: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> AllocResult {
        match self.allocator.lock() {
            Ok(mut allocator) => {
                allocator.reallocate(&Realloc::shrink(address, old_layout, new_layout))
            }
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
            Ok(mut allocator) => {
                allocator.reallocate(&Realloc::grow(address, old_layout, new_layout))
            }
            Err(_) => Err(AllocError),
        }
    }

    unsafe fn grow_zeroed(
        &self,
        address: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> AllocResult {
        let new_address = self.grow(address, old_layout, new_layout)?;
        let zero_from = new_address
            .as_mut_ptr()
            .map_addr(|addr| addr + old_layout.size());
        zero_from.write_bytes(0, new_layout.size() - old_layout.size());

        Ok(new_address)
    }
}

unsafe impl GlobalAlloc for Rulloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match self.allocate(layout) {
            Ok(address) => address.cast().as_ptr(),
            Err(_) => ptr::null_mut(),
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocate(NonNull::new_unchecked(ptr), layout)
    }

    unsafe fn realloc(&self, address: *mut u8, old_layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = Layout::from_size_align(new_size, old_layout.align()).unwrap();
        let address = NonNull::new_unchecked(address);

        let result = if old_layout.size() <= new_size {
            self.shrink(address, old_layout, new_layout)
        } else {
            self.grow(address, old_layout, new_layout)
        };

        match result {
            Ok(new_address) => new_address.as_mut_ptr(),
            Err(_) => ptr::null_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync,
        thread::{self, ThreadId},
    };

    use super::*;
    use crate::platform::PAGE_SIZE;

    #[test]
    fn internal_allocator_wrapper() {
        let allocator = Rulloc::with_default_config();
        unsafe {
            let layout1 = Layout::array::<u8>(8).unwrap();
            let mut addr1 = allocator.allocate(layout1).unwrap();

            addr1.as_mut().fill(69);

            let layout2 = Layout::array::<u8>(PAGE_SIZE * 2).unwrap();
            let mut addr2 = allocator.allocate(layout2).unwrap();

            addr2.as_mut().fill(42);

            for value in addr1.as_ref() {
                assert_eq!(value, &69);
            }

            allocator.deallocate(addr1.cast(), layout1);

            for value in addr2.as_ref() {
                assert_eq!(value, &42);
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

            // Now let's try some reallocs
            let mut realloc_addr = allocator.allocate(layout1).unwrap();
            let corruption_check = 213;
            realloc_addr.as_mut().fill(corruption_check);

            realloc_addr = allocator
                .reallocate(&Realloc::grow(realloc_addr.cast(), layout1, layout2))
                .unwrap();
            verify_number_of_regions_per_bucket!([0, 1, 0]);

            realloc_addr = allocator
                .reallocate(&Realloc::grow(realloc_addr.cast(), layout2, layout3))
                .unwrap();
            verify_number_of_regions_per_bucket!([0, 0, 1]);

            for value in &realloc_addr.as_ref()[..layout1.size()] {
                assert_eq!(*value, corruption_check);
            }
        }
    }

    fn verify_buckets_are_empty(allocator: Rulloc) {
        let internal = allocator.allocator.lock().unwrap();
        for bucket in &internal.buckets {
            assert_eq!(bucket.regions().len(), 0);
        }
        assert_eq!(internal.dyn_bucket.regions().len(), 0);
    }

    /// We'll make all the threads do only allocs at the same time, then wait
    /// and do only deallocs at the same time.
    #[test]
    fn multiple_threads_synchronized_allocs_and_deallocs() {
        let allocator = Rulloc::with_default_config();

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
        let allocator = Rulloc::with_default_config();

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
                    let num_allocs = if cfg!(miri) { 20 } else { 1000 };

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
