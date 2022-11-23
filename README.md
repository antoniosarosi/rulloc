# Memalloc

The code I ended up with after researching memory allocators. You can run
the examples as follows:

```bash
# This one goes step by step, press enter for the next step
cargo run --example bump
# This one requests some memory and prints stats
cargo run --example mmap
```

Run the tests:

```bash
cargo test
```

If you want to start with something really simple, check the source code at
[`src/bump.rs`](./src/bump.rs). The other allocator at
[`src/mmap.rs`](./src/mmap.rs) is much more complicated and implements some
tricks and optimizations, but is well documented. These are the resources I
found the most usefull for writing general purpose allocators:

- [Writing a Memory Allocator - Dimitry Soshinkov](http://dmitrysoshnikov.com/compilers/writing-a-memory-allocator/)
- [Project 3: Memory Allocator - CS 326 USF](https://www.cs.usfca.edu/~mmalensek/cs326/assignments/project-3.html)
- [Memory Allocators 101 - Arjun Sreedharan](https://arjunsreedharan.org/post/148675821737/memory-allocators-101-write-a-simple-memory)
- [`malloc` source code](https://github.com/bminor/glibc/blob/master/malloc/malloc.c)
- [`mmap` linux man page](https://man7.org/linux/man-pages/man2/mmap.2.html)
