use std::{alloc::Layout, mem, ptr};

use libc::{c_void, intptr_t, sbrk};

use crate::align;

// TODO: Implement `alloc::alloc::GloballAlloc` trait. Probably using
// `std::cell::UnsafeCell`?

/// Block header, contains metadata about the block.
struct Header {
    /// Size of the block excluding header.
    size: usize,
    /// Whether the block can be used for a new allocation or is already in use.
    is_free: bool,
    /// Next block.
    next: *mut Header,
}

/// Naive implementation of a bump allocator. Increments the program break if
/// it is not possible to use a previously allocated block. Otherwise it will
/// use the first free block that can hold the requested amount of bytes (first
/// fit algorithm).
pub struct BumpAllocator {
    /// First block.
    first: *mut Header,
    /// Last block.
    last: *mut Header,
}

impl BumpAllocator {
    pub fn new() -> Self {
        Self {
            first: ptr::null_mut(),
            last: ptr::null_mut(),
        }
    }

    /// Returns the first free block that can fit `size` bytes or null pointer
    /// if none was found.
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

    /// Returns a valid pointer on success or null pointer on failure.
    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        // Search free block, if we find one we are done.
        let mut free_block = self.find_free_block(layout.size());
        if !free_block.is_null() {
            (*free_block).is_free = false;
            return (free_block as *mut u8).add(mem::size_of::<Header>());
        }

        // If we didn't find a free block we'll have to create a new one. First,
        // calculate the alignment for Header + content.
        let total_size = align(mem::size_of::<Header>() + layout.size());

        // Increment program break and return null if it fails.
        let address = sbrk(total_size as intptr_t);
        if address == usize::MAX as *mut c_void {
            return ptr::null_mut();
        }

        // Write the header at the beginning of the previous program break
        // address.
        let header = address as *mut Header;
        (*header).is_free = false;
        (*header).next = ptr::null_mut();
        (*header).size = total_size - mem::size_of::<Header>();

        // Update linked list.
        if self.first.is_null() {
            self.first = header;
            self.last = header;
        } else {
            (*self.last).next = header;
            self.last = header;
        }

        // Return a pointer that points right _after_ our header.
        return (address as *mut u8).add(mem::size_of::<Header>());
    }

    /// Deallocates the given pointer.
    pub unsafe fn dealloc(&mut self, pointer: *mut u8) {
        // Substract the header size from the given pointer and interpret that
        // address as a `Header` struct. This causes undefined behaviour if the
        // pointer is not exactly one of the pointers that we issued previously.
        let header = pointer.sub(mem::size_of::<Header>()) as *mut Header;
        (*header).is_free = true;

        // Nothing to do, we can't decrement program break.
        if header != self.last {
            return;
        }

        // This is the last block, so remove it from the linked list.
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

        // Calculate the alignment again to obtain the decrement value.
        let decrement = 0 - align((*header).size + mem::size_of::<Header>()) as isize;
        sbrk(decrement as intptr_t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_allocator() {
        let mut allocator = BumpAllocator::new();

        unsafe {
            // Allocate memory for one item.
            let first_addr = allocator.alloc(Layout::new::<u64>()) as *mut u64;
            *first_addr = 3u64;
            assert_eq!(*first_addr, 3);

            // Allocate memory for an array of items of different type.
            let size: usize = 6;
            let second_addr = allocator.alloc(Layout::array::<u16>(size).unwrap()) as *mut u16;
            for i in 0..size {
                *(second_addr.add(i)) = (i + 1) as u16;
            }

            // Check that nothing has been corrupted.
            assert_eq!(*first_addr, 3);
            for i in 0..size {
                assert_eq!((i + 1) as u16, *(second_addr.add(i)))
            }

            // Deallocate first block.
            allocator.dealloc(first_addr as *mut u8);

            // Allocate memory again for a smaller item, should reuse first block.
            let third_addr = allocator.alloc(Layout::new::<u32>()) as *mut u32;
            assert_eq!(first_addr as *mut u32, third_addr);

            // Deallocate first block again.
            allocator.dealloc(third_addr as *mut u8);

            // Next allocated block cannot reuse the first one as it needs more space.
            let fourth_addr = allocator.alloc(Layout::new::<u128>()) as *mut u128;
            *fourth_addr = 25;

            assert!(fourth_addr > third_addr as *mut u128);
            assert_eq!(*fourth_addr, 25);

            // Deallocate fourth block. Should decrement program break.
            allocator.dealloc(fourth_addr as *mut u8);

            // Now if we allocate something that doesn't fit in the first block
            // the allocator should give as the same address as before.
            let fifth_addr = allocator.alloc(Layout::new::<u128>()) as *mut u128;
            assert_eq!(fifth_addr, fourth_addr);
        }
    }
}
