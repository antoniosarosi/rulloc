# Rulloc

Rulloc (Rust Allocator) is a general purpose memory allocator written in Rust.
It implements
[`std::alloc::Allocator`](https://doc.rust-lang.org/std/alloc/trait.Allocator.html)
and
[`std::alloc::GlobalAlloc`](https://doc.rust-lang.org/stable/std/alloc/trait.GlobalAlloc.html)
traits. All memory is requested from the kernel using
[`mmap`](https://man7.org/linux/man-pages/man2/mmap.2.html) syscalls on Linux
and
[`VirtualAlloc`](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)
on Windows. You can run the examples as follows:

```bash
cargo run --example standalone
cargo run --example global
cargo run --example buckets
cargo run --example aligned
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
cargo miri run --example aligned
```

Global allocator example doesn't work with Miri, see
[`examples/global.rs`](./examples/global.rs).

## Implementation

I started this project for learning purposes, and I know the best way to make
sure you understand something is explaining it to others. So there's plenty of
documentation and ASCII diagrams throughout the codebase if you're interested
in how memory allocators work internally. Start by reading
[`src/allocator.rs`](./src/allocator.rs) for a quick overview and you'll be
redirected to the rest of files through
[intra-doc links](https://doc.rust-lang.org/rustdoc/write-documentation/linking-to-items-by-name.html).

### [Block](./src/block.rs)

If you don't want to scroll through hundreds of lines of code, this is how a
memory block looks like:


```text
+----------------------------+
| pointer to next block      |   <------+
+----------------------------+          |
| pointer to prev block      |          |
+----------------------------+          |
| pointer to block region    |          |
+----------------------------+          | Block Header
| block size                 |          |
+----------------------------+          |
| is free flag (1 byte)      |          |
+----------------------------+          |
| padding (struct alignment) |   <------+
+----------------------------+
|       Block content        |   <------+
|            ...             |          |
|            ...             |          | Addressable content
|            ...             |          |
|            ...             |   <------+
+----------------------------+
```

### [Region](./src/region.rs)

All blocks belong to a memory region, which is a contiguous chunk of memory
that can store multiple pages. In other words, the size of each region is a
multiple of the virtual memory page size on the current platform, and each
region contains a linked list of blocks:

```text
+--------+--------------------------------------------------+
|        | +-------+    +-------+    +-------+    +-------+ |
| Region | | Block | -> | Block | -> | Block | -> | Block | |
|        | +-------+    +-------+    +-------+    +-------+ |
+--------+--------------------------------------------------+
```

### [Bucket](./src/bucket.rs)

The allocator can be configured at compile time with multiple allocation buckets
of different sizes in order to reduce fragmentation. Each bucket stores a linked
list of memory regions and a free list, which is basically a linked list of only
free blocks:

```text
                   Next free block in the free list            Next free block
               +--------------------------------------+   +-----------------------+
               |                                      |   |                       |
+--------+-----|------------------+      +--------+---|---|-----------------------|-----+
|        | +---|---+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
| Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
|        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
+--------+------------------------+      ---------+-------------------------------------+
```

### [Allocator](./src/allocator.rs)

And finally, to put it all together, the allocator is an array of buckets:

```text
                                 Next Free Block                    Next Free Block
                      +------------------------------------+   +-----------------------+
                      |                                    |   |                       |
     +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
     |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
0 -> | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
     |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
     +--------+------------------------+      +--------+-------------------------------------+

                                              Next Free Block
                                 +----------------------------------------+
                                 |                                        |
     +--------+------------------|-----+      +--------+------------------|------------------+
     |        | +-------+    +---|---+ |      |        | +-------+    +---|---+    +-------+ |
1 -> | Region | | Block | -> | Free  | | ---> | Region | | Block | -> | Free  | -> | Block | |
     |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
     +--------+------------------------+      +--------+-------------------------------------+

..............................................................................................

                                        Next Free Block
                                 +---------------------------+
                                 |                           |
     +--------+------------------|-----+      +--------+-----|-------------------------------+
     |        | +-------+    +---|---+ |      |        | +---|---+    +-------+    +-------+ |
N -> | Region | | Block | -> | Free  | | ---> | Region | | Free  | -> | Block | -> | Block | |
     |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
     +--------+------------------------+      +--------+-------------------------------------+
```

All these data structures are a little bit more complicated than that, but
you'll have to read the source code for further details.

## TODO

- [ ] Multithreading. Right now the allocator stands behind a mutex, which is
not the most efficient way of handling multiple threads. We also need better
tests, maybe use [`loom`](https://docs.rs/loom/latest/loom/)? See
[`src/allocator.rs`](./src/allocator.rs) for optimization ideas and low level
details. Here's a quick summary:
  1. One mutex per bucket instead of one global mutex.
  2. Fixed number of allocators and load distribution between threads.
  3. One allocator per thread.

- [ ] Reduce number of system calls for allocations and deallocations. The
allocator requests just enough memory from the OS to store the user content plus
headers, but that could result in calling `mmap` (or equivalent on other
platforms) too many times. In order to avoid this, we could figure out how many
extra pages to request *in general*, for example if we need 1 page to store our
stuff plus user's stuff, we can request 10 pages instead and avoid kernel code
execution for further allocations.

- [ ] Study if it is possible to reduce the size of our headers. On 64 bit
machines the size of each block header is 40 bytes and the size of each region
header is 48 bytes. Region headers are not so important, but block headers
are important for small allocations.

- [ ] Memory alignment optimization. The current implementation adds padding
to the beginning of the block content in order to align pointers. This is only
done when the alignment constraint is greater than the pointer size on the
current machine, and it's not an issue for *small* alignments such as 16 or 32,
but it might waste too much space for alignments of 512 and beyond. One possible
starting point for this issue: memory regions are already aligned to page size
(usually 4096), so we could probably take advantage of that somehow.

- [ ] Free blocks searching algorithm optimization. We do have a free list,
which makes finding free blocks easier, and we also have buckets of fixed sizes,
so this isn't particularly *unoptimized*. Except for sequential large
allocations of different sizes. Whenever we receive an allocation request that
we can't fit in any bucket because it's too large, we use the *Dynamic Bucket*,
which is just a fancy name for a bucket that doesn't care about sizes and can
store blocks of 1KB mixed with blocks of 1GB. For that bucket at least, we could
implement a heap instead of a normal linked list for searching free blocks, or
maybe just store a pointer to the biggest block so that we know immediately if
we need to bother the kernel or not.

- [ ] What should we do when `mmap` or `VirtualAlloc` fails? Panic? The
allocator user doesn't know anything about the failure because
[`std::alloc::Allocator`](https://doc.rust-lang.org/std/alloc/trait.Allocator.html)
doesn't have a return type for deallocations. Panicking doesn't seem the way to
go because the program can continue normally, but how and when are we returning
the memory region to the kernel then?

- [ ] Remove
[`std::alloc::GlobalAlloc`](https://doc.rust-lang.org/stable/std/alloc/trait.GlobalAlloc.html)
implementation when
[`std::alloc::Allocator`](https://doc.rust-lang.org/std/alloc/trait.Allocator.html)
becomes stable.
