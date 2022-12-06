#![feature(allocator_api)]
#![feature(is_some_and)]
#![feature(nonnull_slice_from_raw_parts)]

mod mmap;

pub use mmap::MmapAllocator;
