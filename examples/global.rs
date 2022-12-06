use memalloc::MmapAllocator;

#[global_allocator]
static ALLOCATOR: MmapAllocator = MmapAllocator::with_default_config();

fn main() {
    let num = Box::new(10);
    println!("Boxed num {num}");

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

    let mut vec: Vec<u8> = Vec::with_capacity(1024 * 10);
    vec.push(1);

    println!("Large allocation at {:?}", vec.as_ptr());
}
