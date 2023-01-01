//! General purpose memory allocator. Memory regions are requested from the
//! underlying kernel using
//! [`mmap`](https://man7.org/linux/man-pages/man2/mmap.2.html) syscalls on
//! Unix platforms and
//! [`VirtualAlloc`](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc)
//! on Windows platforms. This is how the allocator looks like internally:
//!
//! ```text
//!                                  Next Free Block                    Next Free Block
//!                       +------------------------------------+   +-----------------------+
//!                       |                                    |   |                       |
//!      +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
//!      |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
//! 0 -> | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
//!      |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
//!      +--------+------------------------+      +--------+-------------------------------------+
//!
//!                                               Next Free Block
//!                                  +----------------------------------------+
//!                                  |                                        |
//!      +--------+------------------|-----+      +--------+------------------|------------------+
//!      |        | +-------+    +---|---+ |      |        | +-------+    +---|---+    +-------+ |
//! 1 -> | Region | | Block | -> | Free  | | ---> | Region | | Block | -> | Free  | -> | Block | |
//!      |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
//!      +--------+------------------------+      +--------+-------------------------------------+
//!
//! ..............................................................................................
//!
//!                                         Next Free Block
//!                                  +---------------------------+
//!                                  |                           |
//!      +--------+------------------|-----+      +--------+-----|-------------------------------+
//!      |        | +-------+    +---|---+ |      |        | +---|---+    +-------+    +-------+ |
//! N -> | Region | | Block | -> | Free  | | ---> | Region | | Free  | -> | Block | -> | Block | |
//!      |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
//!      +--------+------------------------+      +--------+-------------------------------------+
//! ```
//!
//! The allocator contains multiple buckets, each bucket contains a list of
//! regions and each region stores a list of memory blocks. Implemented
//! optimizations:
//!
//! - **Block coalescing**: merge adjacent free blocks into one bigger block.
//! - **Block splitting**: blocks that are too big are split in two blocks.
//! - **Free list**: linked list of only free blocks, finds free blocks faster.
//! - **In place reallocations**: if possible, avoid creating new memory blocks.
//! - **Fixed size buckets**: reduce fragmentation by grouping allocation sizes.
//!
//! See [`Rulloc`] for usage examples.

#![feature(allocator_api)]
#![feature(alloc_layout_extra)]
#![feature(is_some_and)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(strict_provenance)]
#![feature(slice_ptr_get)]

use std::{alloc::AllocError, ptr::NonNull};

mod alignment;
mod allocator;
mod block;
mod bucket;
mod freelist;
mod header;
mod list;
mod platform;
mod realloc;
mod region;

/// Non-null pointer to `T`. We use this in most cases instead of `*mut T`
/// because the compiler will yell at us if we don't write code for the `None`
/// case. I think variance doesn't have much implications here except for
/// [`list::LinkedList<T>`], but that should probably be covariant anyway.
pub(crate) type Pointer<T> = Option<NonNull<T>>;

/// Shorter syntax for allocation/reallocation return types.
pub(crate) type AllocResult = Result<NonNull<[u8]>, AllocError>;

pub use allocator::Rulloc;
