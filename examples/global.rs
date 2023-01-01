use rulloc::Rulloc;

// NOTE: This example doesn't work with Miri because we use the global alocator
// to simulate `mmap` calls when `cfg!(miri)`. If we are the global allocator,
// there are two problems:
//
// 1. We cannot do FFI calls when using Miri, so no `mmap`. That's why we
// simulate them with `std::alloc::alloc`.
//
// 2. The allocator needs to acquire a `Mutex` lock in order to modify its
// structures. Whenever somebody calls `allocate` on our allocator (for example
// `Box`) the calling thread will acquire the lock. However, when the thread
// calls `mmap` to obtain a new region, `std::alloc::alloc` will be called
// instead, but WE are the allocator, so we'll try to acquire the lock again,
// which causes a deadlock. It turns out that we cannot simulate ourselves
// within ourselves :(

#[global_allocator]
static ALLOCATOR: Rulloc = Rulloc::with_default_config();

fn main() {
    let num = Box::new(10);
    println!("Boxed num {num} at {:?}", &*num as *const usize);

    let mut vec = Vec::with_capacity(*num);

    for i in 0..*num {
        vec.push(i);
    }

    println!("Vec: {vec:?} at {:?}", vec.as_ptr());

    let handle = std::thread::spawn(|| {
        let mut vec: Vec<u8> = Vec::with_capacity(256);
        vec.push(5);
        vec.push(6);
        println!("Second thread Vec: {vec:?} at {:?}", vec.as_ptr());
    });

    handle.join().unwrap();

    let cap = 1024 * 1024;
    let mut vec: Vec<u8> = Vec::with_capacity(cap);
    vec.push(1);

    println!("Large allocation of {cap} bytes at {:?}", vec.as_ptr());
}
