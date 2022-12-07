# Memalloc

Memory allocator written in Rust. It implements
[`std::alloc::Allocator`](https://doc.rust-lang.org/std/alloc/trait.Allocator.html)
and [`std::alloc::GlobalAlloc`](https://doc.rust-lang.org/stable/std/alloc/trait.GlobalAlloc.html)
traits. All memory is requested from the kernel using `mmap` syscalls. You can
run the examples as follows:

```bash
cargo run --example standalone
cargo run --example global
cargo run --example buckets
```

Run the tests:

```bash
cargo test
```

Run with [Miri](https://github.com/rust-lang/miri):

```bash
cargo miri test
cargo miri run --example standalone
cargo miri run --example buckets
```

Global allocator example doesn't work with Miri, see
[`./examples/global.rs`](./examples/global.rs).
