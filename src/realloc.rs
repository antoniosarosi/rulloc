use std::{alloc::Layout, ptr::NonNull};

use crate::{block::Block, header::Header};

/// This is used to carry realloc metadata around the code as we don't want to
/// pass the same parameters over and over again. For the real reallocation
/// business, see [`crate::bucket`] and [`crate::allocator`].
pub(crate) struct Realloc {
    /// Source block, or the block whose contents we are trying to reallocate.
    pub block: NonNull<Header<Block>>,
    /// Current user pointer. This is located somewhere in `block`, depends on
    /// padding and alignment.
    pub address: NonNull<u8>,
    /// The layout that the new allocation should fit.
    pub new_layout: Layout,
    /// Layout of the previous allocation.
    pub old_layout: Layout,
    /// For now, whether shrink or grow.
    pub method: ReallocMethod,
}

/// The reallocation either attempts to shrink or grow, but there are some
/// details. The trait [`std::alloc::Allocator`] provides the methods
/// [`std::alloc::Allocator::grow`] and [`std::alloc::Allocator::shrink`],
/// but both of those accept new layouts with the same size as the previously
/// allocated layout. If we want to know exactly what the user is attempting,
/// we better store that info.
#[derive(Clone, Copy)]
pub(crate) enum ReallocMethod {
    Shrink,
    Grow,
}

impl Realloc {
    /// Builds a new [`Realloc`] with the given parameters. Caller must
    /// ensure that `address` is valid because we'll attempt to obtain the
    /// block where `address` is located.
    pub unsafe fn new(
        address: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
        method: ReallocMethod,
    ) -> Self {
        let block = Header::<Block>::from_allocated_pointer(address, old_layout);

        Self {
            block,
            address,
            new_layout,
            old_layout,
            method,
        }
    }

    /// Shorter syntax for building a new [`Realloc`] with [`ReallocMethod::Shrink`].
    pub unsafe fn shrink(address: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Self {
        Self::new(address, old_layout, new_layout, ReallocMethod::Shrink)
    }

    /// Shorter syntax for building a new [`Realloc`] with [`ReallocMethod::Grow`].
    pub unsafe fn grow(address: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Self {
        Self::new(address, old_layout, new_layout, ReallocMethod::Grow)
    }

    /// Number of bytes that should be copied from the previous allocation. If
    /// we are shrinking, we only need to copy enough bytes to fill the new
    /// layout, otherwise we'll copy everything from the previous layout.
    pub fn count(&self) -> usize {
        match self.method {
            ReallocMethod::Shrink => self.new_layout.size(),
            ReallocMethod::Grow => self.old_layout.size(),
        }
    }

    /// Maps this [`Realloc`] to a [`Realloc`] on a new block. This is usefull
    /// for growing, see [`crate::bucket`].
    pub unsafe fn map(&self, block: NonNull<Header<Block>>) -> Self {
        Self { block, ..*self }
    }
}
