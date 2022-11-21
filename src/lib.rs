use std::{alloc::Layout, mem, ptr};

use libc::{c_void, intptr_t, sbrk};

struct Header {
    size: usize,
    is_free: bool,
    next: *mut Header,
}

pub struct BumpAllocator {
    first: *mut Header,
    last: *mut Header,
}

fn align(to_be_aligned: usize) -> usize {
    (to_be_aligned + mem::size_of::<usize>() - 1) & !(mem::size_of::<usize>() - 1)
}

impl BumpAllocator {
    pub fn new() -> Self {
        Self {
            first: ptr::null_mut(),
            last: ptr::null_mut(),
        }
    }

    unsafe fn find_free_block(&self, size: usize) -> *mut Header {
        let mut current = self.first;

        while !current.is_null() {
            if (*current).size >= size && (*current).is_free {
                return current;
            }
            current = (*current).next;
        }

        ptr::null_mut()
    }

    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let mut free_block = self.find_free_block(layout.size());

        if !free_block.is_null() {
            (*free_block).is_free = false;
            return (free_block as *mut u8).add(mem::size_of::<Header>());
        }

        let total_size = mem::size_of::<Header>() + layout.size();

        let address = sbrk(align(total_size) as intptr_t);

        if address == usize::MAX as *mut c_void {
            return ptr::null_mut();
        }

        let header = address as *mut Header;
        (*header).is_free = false;
        (*header).next = ptr::null_mut();
        (*header).size = layout.size();

        if self.first.is_null() {
            self.first = header;
            self.last = header;
        } else {
            (*self.last).next = header;
            self.last = header;
        }

        return (address as *mut u8).add(mem::size_of::<Header>());
    }

    pub unsafe fn dealloc(&mut self, pointer: *mut u8) {
        let header = pointer.sub(mem::size_of::<Header>()) as *mut Header;
        (*header).is_free = true;

        if header != self.last {
            return;
        }

        if self.first == self.last {
            self.first = ptr::null_mut();
            self.last = ptr::null_mut();
        } else {
            let mut current = self.first;
            while !(*current).next.is_null() && (*current).next != self.last {
                current = (*current).next;
            }
            self.last = current;
        }

        let decrement = 0 - align((*header).size + mem::size_of::<Header>());
        sbrk(decrement as intptr_t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align() {
        let ptr_size = mem::size_of::<usize>();

        let mut alignments = Vec::new();

        for i in 0..10 {
            let sizes = (ptr_size * i + 1)..=(ptr_size * (i + 1));
            let expected_alignment = ptr_size * (i + 1);
            alignments.push((sizes, expected_alignment));
        }

        for (sizes, expected) in alignments {
            for size in sizes {
                println!("{} {}", size, expected);
                assert_eq!(expected, align(size));
            }
        }
    }

    #[test]
    fn test_alloc() {
        let mut allocator = BumpAllocator::new();

        unsafe {
            // Allocate memory for one item
            let first_addr = allocator.alloc(Layout::new::<u64>()) as *mut u64;
            *first_addr = 3u64;
            assert_eq!(*first_addr, 3);
            // Allocate memory for an array of items of different type
            let size: usize = 6;
            let second_addr = allocator.alloc(Layout::array::<u16>(size).unwrap()) as *mut u16;
            for i in 0..size {
                *(second_addr.add(i)) = (i + 1) as u16;
            }

            // Check that nothing has been corrupted
            assert_eq!(*first_addr, 3);
            for i in 0..size {
                assert_eq!((i + 1) as u16, *(second_addr.add(i)))
            }

            // Deallocate first block
            allocator.dealloc(first_addr as *mut u8);

            // Allocate memory again for a smaller item, should reuse first block
            let third_addr = allocator.alloc(Layout::new::<u32>()) as *mut u32;
            assert_eq!(first_addr as *mut u32, third_addr);

            // Deallocate first block again
            allocator.dealloc(third_addr as *mut u8);

            // Next allocated block cannot reuse the first one as it needs more space
            let fourth_addr = allocator.alloc(Layout::new::<u128>()) as *mut u128;
            *fourth_addr = 25;

            assert!(fourth_addr > third_addr as *mut u128);
            assert_eq!(*fourth_addr, 25);
        }
    }
}
