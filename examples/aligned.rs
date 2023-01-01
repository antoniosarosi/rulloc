#![feature(allocator_api)]

use std::{
    alloc::{Allocator, Layout},
    ptr::NonNull,
};

use rulloc::Rulloc;

fn print_alloc(addr: NonNull<u8>, layout: Layout) {
    println!(
        "\nAlloc of size {} and alignment {} at {addr:?}",
        layout.size(),
        layout.align()
    );
    println!(
        "Alignment check: {addr:?} % {} = {}",
        layout.align(),
        addr.as_ptr() as usize % layout.align()
    );
}

fn main() {
    let allocator = Rulloc::default();

    unsafe {
        let layout1 = Layout::from_size_align(1, 16).unwrap();
        let addr1 = allocator.allocate(layout1).unwrap().cast();
        print_alloc(addr1, layout1);

        let layout2 = Layout::from_size_align(1, 32).unwrap();
        let addr2 = allocator.allocate(layout2).unwrap().cast();
        print_alloc(addr2, layout2);

        let layout3 = Layout::from_size_align(1, 1024).unwrap();
        let addr3 = allocator.allocate(layout3).unwrap().cast();
        print_alloc(addr3, layout3);

        allocator.deallocate(addr1, layout1);
        allocator.deallocate(addr2, layout2);
        allocator.deallocate(addr3, layout3);
    }
}
