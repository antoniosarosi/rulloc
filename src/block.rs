use std::{alloc::Layout, mem, ptr::NonNull};

use crate::{alignment, freelist::FreeListNode, header::Header, region::Region};

/// Minimum block size in bytes. Read the documentation of the data structures
/// used for this allocator to understand why it has this value. Specifically,
/// see [`Header<T>`], [`Block`], [`Region`], [`crate::list::LinkedList<T>`] and
/// especially [`FreeListNode`].
pub(crate) const MIN_BLOCK_SIZE: usize = mem::size_of::<FreeListNode>();

/// Block header size in bytes. See [`Header<T>`] and [`Block`].
pub(crate) const BLOCK_HEADER_SIZE: usize = mem::size_of::<Header<Block>>();

/// Memory block specific data. All headers are also linked list nodes, see
/// [`Header<T>`]. In this case, a complete block header would be
/// [`crate::list::Node<Block>`], also known as [`Header<Block>`]. Here's a
/// graphical representation of how it looks like in memory:
///
/// ```text
/// +----------------------------+          <----------------------+
/// | pointer to next block      |   <------+                      |
/// +----------------------------+          | Pointer<Node<Block>> |
/// | pointer to prev block      |   <------+                      |
/// +----------------------------+                                 |
/// | pointer to block region    |   <------+                      |
/// +----------------------------+          |                      | <Node<Block>>
/// | block size                 |          |                      |
/// +----------------------------+          | Block                |
/// | is free flag (1 byte)      |          |                      |
/// +----------------------------+          |                      |
/// | padding (struct alignment) |   <------+                      |
/// +----------------------------+          <----------------------+
/// |       Block content        |   <------+
/// |            ...             |          |
/// |            ...             |          | Addressable content
/// |            ...             |          |
/// |            ...             |   <------+
/// +----------------------------+
/// ```
///
/// Note that the order of struct fields doesn't matter, this is just an
/// example. The compiler might reorder the fields in a different way unless we
/// use [`repr`](https://doc.rust-lang.org/nomicon/repr-rust.html), but we
/// never assume any specific order on the struct fields, so we don't need it.
///
/// The block content is where the allocator users write their data. However,
/// the pointer that we provide them with **MAY NOT** point exactly to the first
/// address of the block content. That's because we have to support alignments
/// of any size (any power of 2), so we need to make sure that the pointers we
/// return satisfy the required alignment constraints. See [`crate::alignment`]
/// for a detailed explanation. Also note that if a block is free (not
/// currently used by the caller) we take advantage of the fact that we can
/// put anything we want in the block content. See [`crate::freelist`].
pub(crate) struct Block {
    /// Memory region where this block is located.
    pub region: NonNull<Header<Region>>,
    /// Size of the block excluding [`Header<Block>`] size.
    pub size: usize,
    /// Whether this block can be used or not.
    pub is_free: bool,
}

impl Header<Block> {
    /// Returns the actual block header of a free list node. See
    /// [`FreeListNode`].
    ///
    /// # Safety
    ///
    /// This function is only used internally, we manage the free list so we can
    /// guarantee that we won't call this function with an invalid address. This
    /// is only unsafe if the allocator user writes to an address that was
    /// previously deallocated (use after free).
    #[inline]
    pub unsafe fn from_free_list_node(links: NonNull<FreeListNode>) -> NonNull<Self> {
        Self::from_content_address(links.cast())
    }

    /// See [`crate::alignment::AlignmentBackPointer`].
    #[inline]
    pub unsafe fn from_aligned_address(address: NonNull<u8>) -> NonNull<Self> {
        *alignment::back_pointer_of(address).as_ptr()
    }

    /// See [`Header::from_content_address`] and [`Self::from_aligned_address`].
    #[inline]
    pub unsafe fn from_allocated_pointer(address: NonNull<u8>, layout: Layout) -> NonNull<Self> {
        if layout.align() <= alignment::POINTER_SIZE {
            Self::from_content_address(address)
        } else {
            Self::from_aligned_address(address)
        }
    }

    /// Returns a mutable reference to the region that contains this block.
    #[inline]
    pub unsafe fn region_mut(&mut self) -> &mut Header<Region> {
        self.data.region.as_mut()
    }

    /// Helper function to reduce boilerplate. Since the complete block header
    /// is [`Header<Block>`] all [`Block`] fields have to be accessed through
    /// `data`.
    #[inline]
    pub fn is_free(&self) -> bool {
        self.data.is_free
    }

    /// Block size excluding [`BLOCK_HEADER_SIZE`].
    #[inline]
    pub fn size(&self) -> usize {
        self.data.size
    }

    /// Total block size including [`BLOCK_HEADER_SIZE`].
    #[inline]
    pub fn total_size(&self) -> usize {
        BLOCK_HEADER_SIZE + self.data.size
    }
}
