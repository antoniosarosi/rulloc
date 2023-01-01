#![feature(allocator_api)]

use std::{
    alloc::{Allocator, Layout},
    ptr::NonNull,
};

use rulloc::Rulloc;

fn print_alloc(addr: NonNull<u8>, layout: Layout) {
    println!("Requested {} bytes of memory", layout.size());
    println!("Received this address: {addr:?}");
}

fn main() {
    let allocator = Rulloc::default();

    unsafe {
        let layout1 = Layout::new::<u8>();
        let addr1 = allocator.allocate(layout1).unwrap().cast();
        print_alloc(addr1, layout1);

        let layout2 = Layout::array::<u8>(1024).unwrap();
        let addr2 = allocator.allocate(layout2).unwrap().cast();
        print_alloc(addr2, layout2);

        let layout3 = Layout::array::<u8>(4096).unwrap();
        let addr3 = allocator.allocate(layout3).unwrap().cast();
        print_alloc(addr3, layout3);

        println!("Deallocating everything...");
        allocator.deallocate(addr1, layout1);
        allocator.deallocate(addr2, layout2);
        allocator.deallocate(addr3, layout3);

        println!("\nNow let's try to use Vec with our allocator...");
        let initial_size = 10;
        let mut v = Vec::with_capacity_in(initial_size, allocator);
        for i in 0..initial_size {
            v.push(i as u32);
        }
        println!("Initial Vec at {:?} = {:?}", v.as_ptr(), v);

        println!("Let's try some reallocs");
        let next_chunk_size = 1024;
        for i in 0..next_chunk_size {
            v.push(i);
        }
        println!(
            "Pushed {} elements into vec, current addr = {:?}",
            next_chunk_size,
            v.as_ptr()
        );

        println!("First 15 elements: {:?}", &v[0..15]);
        println!("Last 15 elements: {:?}", &v[v.len() - 15..]);
    }
}
