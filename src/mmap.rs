use std::{alloc::Layout, mem, ptr};

use libc::{self, c_void, mmap, munmap, size_t};

use crate::align;

// TODO: Make it thread local var
#[inline]
unsafe fn page_size() -> usize {
    libc::sysconf(libc::_SC_PAGE_SIZE) as usize
}

pub struct MmapAllocator;

impl MmapAllocator {
    pub fn new() -> Self {
        Self
    }

    pub unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let total_size = align(layout.size());

        let addr = ptr::null_mut::<c_void>();
        let len = (total_size + page_size() - 1) / page_size();
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

        match mmap(addr, len, prot, flags, -1, 0) {
            libc::MAP_FAILED => ptr::null_mut::<u8>(),
            address => address as *mut u8,
        }
    }

    pub unsafe fn dealloc(&self, address: *mut u8, layout: Layout) {
        if munmap(address as *mut c_void, layout.size() as size_t) != 0 {
            // TODO: What should be done here? Panic?
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc() {
        let allocator = MmapAllocator::new();

        unsafe {
            let first_addr = allocator.alloc(Layout::new::<u64>()) as *mut u64;
            *first_addr = 100;

            let second_addr = allocator.alloc(Layout::new::<u64>()) as *mut u64;
            *second_addr = 200;

            assert_eq!(*first_addr, 100);
            assert_eq!(*second_addr, 200);
            assert_ne!(first_addr, second_addr);
        }
    }
}
