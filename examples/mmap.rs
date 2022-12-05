use std::alloc::Layout;

use memalloc::MmapAllocator;

fn print_alloc(size: usize, addr: *mut u8) {
    println!("Requested {size} bytes of memory");
    println!("Received this address: {addr:?}");
}

fn main() {
    let mut allocator = MmapAllocator::new();

    unsafe {
        let addr1 = allocator.alloc(Layout::new::<u8>());
        print_alloc(1, addr1);

        let addr2 = allocator.alloc(Layout::array::<u8>(1024).unwrap());
        print_alloc(1024, addr2);

        let addr3 = allocator.alloc(Layout::array::<u8>(4096).unwrap());
        print_alloc(4096, addr3);

        allocator.dealloc(addr1);
        allocator.dealloc(addr2);
        allocator.dealloc(addr3);
    }
}
