use std::{alloc::Layout, io::Read};

use libc::sbrk;
use memalloc::BumpAllocator;

fn block_until_enter_pressed() {
    std::io::stdin().bytes().next();
}

unsafe fn print_alloc(layout: Layout, addr: *mut u8) {
    println!(
        "Allocated {} bytes, address = {:?}, program break = {:?}",
        layout.size(),
        addr,
        sbrk(0)
    );
}

fn main() {
    let mut allocator = BumpAllocator::new();

    unsafe {
        println!("PID = {}, Program break = {:?}", std::process::id(), sbrk(0));
        block_until_enter_pressed();

        // Allocate space for unsigned 32 bit integer (4 bytes)
        let layout = Layout::new::<u32>();
        let first_block = allocator.alloc(layout);
        print_alloc(layout, first_block);
        block_until_enter_pressed();

        // Allocate 12 bytes
        let layout = Layout::array::<u8>(12).unwrap();
        let second_block = allocator.alloc(layout);
        print_alloc(layout, second_block);
        block_until_enter_pressed();

        // Deallocate first block
        allocator.dealloc(first_block);
        println!("Deallocated block at address = {:?}", first_block);
        block_until_enter_pressed();

        // Check that first block is reused
        let layout = Layout::array::<u8>(2).unwrap();
        let third_block = allocator.alloc(layout);
        print_alloc(layout, third_block);

        block_until_enter_pressed();
    }
}
