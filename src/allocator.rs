use std::{
    alloc::{AllocError, Allocator, GlobalAlloc, Layout},
    cell::UnsafeCell,
    ptr::{self, NonNull},
    sync::RwLock,
};

use crate::bucket::Bucket;

/// This is the main allocator, but it has to be wrapped in [`UnsafeCell`] to
/// satisfy [`std::alloc::Allocator`] trait. See [`MmapAllocator`].
struct InternalAllocator<const N: usize> {
    sizes: [usize; N],
    /// Fixed size buckets.
    buckets: [Bucket; N],
    /// Any allocation request of size > sizes[N - 1] will use this bucket.
    dyn_bucket: Bucket,
}

impl<const N: usize> InternalAllocator<N> {
    pub const fn with_bucket_sizes(sizes: [usize; N]) -> Self {
        InternalAllocator::<N> {
            sizes,
            buckets: [Bucket::new(); N],
            dyn_bucket: Bucket::new(),
        }
    }

    fn dispatch(&mut self, layout: Layout) -> &mut Bucket {
        for (i, bucket) in self.buckets.iter_mut().enumerate() {
            if layout.size() <= self.sizes[i] {
                return bucket;
            }
        }

        &mut self.dyn_bucket
    }

    pub unsafe fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.dispatch(layout).allocate(layout)
    }

    pub unsafe fn deallocate(&mut self, address: NonNull<u8>, layout: Layout) {
        self.dispatch(layout).deallocate(address, layout)
    }
}

/// General purpose allocator. All memory is requested from the kernel using
/// [`libc::mmap`] and some tricks and optimizations are implemented such as
/// free list, block coalescing and block splitting.
pub struct MmapAllocator<const N: usize = 3> {
    allocator: RwLock<UnsafeCell<InternalAllocator<N>>>,
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

unsafe impl<const N: usize> Sync for MmapAllocator<N> {}

impl Default for MmapAllocator {
    fn default() -> Self {
        MmapAllocator::with_default_config()
    }
}

impl MmapAllocator {
    pub const fn with_default_config() -> Self {
        Self {
            allocator: RwLock::new(UnsafeCell::new(InternalAllocator::with_bucket_sizes([
                128, 1024, 8192,
            ]))),
        }
    }
}

impl<const N: usize> MmapAllocator<N> {
    pub fn with_bucket_sizes(sizes: [usize; N]) -> Self {
        Self {
            allocator: RwLock::new(UnsafeCell::new(InternalAllocator::with_bucket_sizes(sizes))),
        }
    }
}

unsafe impl<const N: usize> Allocator for MmapAllocator<N> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            match self.allocator.write() {
                Ok(allocator) => (*allocator.get()).allocate(layout),
                Err(_) => Err(AllocError),
            }
        }
    }

    unsafe fn deallocate(&self, address: NonNull<u8>, layout: Layout) {
        if let Ok(allocator) = self.allocator.write() {
            (*allocator.get()).deallocate(address, layout)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync, thread};

    use crate::{region::{REGION_HEADER_SIZE, PAGE_SIZE}, block::{MIN_BLOCK_SIZE, BLOCK_HEADER_SIZE}};

    use super::*;


    #[test]
    fn wrapper_works() {
        let allocator = MmapAllocator::with_default_config();
        unsafe {
            let layout1 = Layout::array::<u8>(8).unwrap();
            let mut address1 = allocator.allocate(layout1).unwrap();
            let slice1 = address1.as_mut();

            slice1.fill(69);

            let layout2 = Layout::array::<u8>(PAGE_SIZE * 2).unwrap();
            let mut address2 = allocator.allocate(layout2).unwrap();
            let slice2 = address2.as_mut();

            slice2.fill(69);

            for value in slice1 {
                assert_eq!(value, &69);
            }

            allocator.deallocate(address1.cast(), layout1);

            for value in slice2 {
                assert_eq!(value, &69);
            }

            allocator.deallocate(address2.cast(), layout2);
        }
    }

    #[test]
    fn buckets_work() {
        unsafe {
            let mut allocator = InternalAllocator::<3>::with_bucket_sizes([8, 16, 24]);

            let layout1 = Layout::new::<u8>();
            let addr1 = allocator.allocate(layout1).unwrap().cast();

            assert_eq!(allocator.buckets[0].regions.len, 1);
            assert_eq!(allocator.buckets[1].regions.len, 0);
            assert_eq!(allocator.buckets[2].regions.len, 0);

            let layout2 = Layout::array::<u8>(16).unwrap();
            let addr2 = allocator.allocate(layout2).unwrap().cast();

            assert_eq!(allocator.buckets[0].regions.len, 1);
            assert_eq!(allocator.buckets[1].regions.len, 1);
            assert_eq!(allocator.buckets[2].regions.len, 0);

            let layout3 = Layout::array::<u8>(20).unwrap();
            let addr3 = allocator.allocate(layout3).unwrap().cast();

            assert_eq!(allocator.buckets[0].regions.len, 1);
            assert_eq!(allocator.buckets[1].regions.len, 1);
            assert_eq!(allocator.buckets[2].regions.len, 1);

            allocator.deallocate(addr1, layout1);

            assert_eq!(allocator.buckets[0].regions.len, 0);
            assert_eq!(allocator.buckets[1].regions.len, 1);
            assert_eq!(allocator.buckets[2].regions.len, 1);

            allocator.deallocate(addr2, layout2);

            assert_eq!(allocator.buckets[0].regions.len, 0);
            assert_eq!(allocator.buckets[1].regions.len, 0);
            assert_eq!(allocator.buckets[2].regions.len, 1);

            allocator.deallocate(addr3, layout3);

            assert_eq!(allocator.buckets[0].regions.len, 0);
            assert_eq!(allocator.buckets[1].regions.len, 0);
            assert_eq!(allocator.buckets[2].regions.len, 0);

            let layout4 = Layout::array::<u8>(32).unwrap();
            let addr4 = allocator.allocate(layout4).unwrap().cast();

            assert_eq!(allocator.buckets[0].regions.len, 0);
            assert_eq!(allocator.buckets[1].regions.len, 0);
            assert_eq!(allocator.buckets[2].regions.len, 0);
            assert_eq!(allocator.dyn_bucket.regions.len, 1);

            allocator.deallocate(addr4, layout4);
        }
    }

    #[test]
    fn multiple_threads_basic_checks() {
        let allocator = MmapAllocator::with_default_config();

        let num_threads: u8 = 8;

        let barrier = sync::Barrier::new(8);

        thread::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| unsafe {
                    let layout = Layout::array::<thread::ThreadId>(1024).unwrap();
                    let addr = allocator.allocate(layout).unwrap().cast();
                    let id = thread::current().id();
                    for _ in 0..layout.size() {
                        *addr.as_ptr() = id;
                    }

                    barrier.wait();

                    for _ in 0..layout.size() {
                        assert_eq!(*addr.as_ptr(), id);
                    }

                    allocator.deallocate(addr.cast(), layout);
                });
            }
        });

        unsafe {
            let internal = allocator.allocator.read().unwrap().get();
            for bucket in &(*internal).buckets {
                assert_eq!(bucket.regions.len, 0);
            }
            assert_eq!((*internal).dyn_bucket.regions.len, 0);
        }
    }

    #[test]
    fn multiple_threads_allocating_and_deallocating() {
        let allocator = MmapAllocator::with_default_config();

        let num_threads: u8 = 8;

        thread::scope(|scope| {
            for _ in 0..num_threads {
                scope.spawn(|| unsafe {
                    let layout = Layout::array::<u8>(1024).unwrap();

                    for _ in 0..10 {
                        let address = allocator.allocate(layout).unwrap().cast();
                        allocator.deallocate(address, layout);
                    }
                });
            }
        });

        unsafe {
            let internal = allocator.allocator.read().unwrap().get();
            for bucket in &(*internal).buckets {
                assert_eq!(bucket.regions.len, 0);
            }
            assert_eq!((*internal).dyn_bucket.regions.len, 0);
        }
    }
}
