#![feature(allocator_api)]

use std::{
    alloc::{Allocator, Layout},
    ptr::NonNull,
};

use rulloc::Rulloc;

fn print_alloc(addr: NonNull<u8>, layout: Layout) {
    println!("Alloc of {} bytes at {addr:?}", layout.size());
}

fn main() {
    let allocator = Rulloc::<3>::with_bucket_sizes([8, 16, 24]);

    println!("Allocator configured with bucket sizes 8, 16 and 24.");
    println!("Notice how addresses are located in different regions.");
    println!("If page size is 4096 bytes there should be 4KB of difference between them:");

    unsafe {
        let layout1 = Layout::array::<u8>(8).unwrap();
        let addr1 = allocator.allocate(layout1).unwrap().cast();
        print_alloc(addr1, layout1);

        let layout2 = Layout::array::<u8>(16).unwrap();
        let addr2 = allocator.allocate(layout2).unwrap().cast();
        print_alloc(addr2, layout2);

        let layout3 = Layout::array::<u8>(24).unwrap();
        let addr3 = allocator.allocate(layout3).unwrap().cast();
        print_alloc(addr3, layout3);

        allocator.deallocate(addr1, layout1);
        allocator.deallocate(addr2, layout2);
        allocator.deallocate(addr3, layout3);
    }
}
