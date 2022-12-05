#![feature(allocator_api)]
#![feature(is_some_and)]

mod align;
mod bump;
mod mmap;

pub use align::align;
pub use bump::BumpAllocator;
pub use mmap::MmapAllocator;
