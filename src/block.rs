use std::{mem, ptr::NonNull};

use crate::{freelist::FreeListNode, header::Header, region::Region};

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
/// +--------------------------+          <----------------------+
/// | pointer to next block    |   <------+                      |
/// +--------------------------+          | Pointer<Node<Block>> |
/// | pointer to prev block    |   <------+                      |
/// +--------------------------+                                 |
/// | pointer to block region  |   <------+                      |
/// +--------------------------+          |                      | <Node<Block>>
/// | block size               |          |                      |
/// +--------------------------+          | Block                |
/// | is free flag             |          |                      |
/// +--------------------------+          |                      |
/// | padding (word alignment) |   <------+                      |
/// +--------------------------+          <----------------------+
/// |      User content        |   <------+
/// |           ...            |          |
/// |           ...            |          | This is where the user writes stuff.
/// |           ...            |          |
/// |           ...            |   <------+
/// +--------------------------+
/// ```
#[derive(Clone, Copy, Debug)]
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
    pub unsafe fn from_free_list_node(links: NonNull<FreeListNode>) -> NonNull<Self> {
        Self::from_content_address(links.cast())
    }

    /// Returns a mutable reference to the region that contains this block.
    pub unsafe fn region_mut(&mut self) -> &mut Header<Region> {
        self.data.region.as_mut()
    }

    /// Helper function to reduce boilerplate. Since the complete block header
    /// is [`Header<Block>`] all [`Block`] fields have to be accessed through
    /// `data`.
    pub fn is_free(&self) -> bool {
        self.data.is_free
    }

    /// Block size excluding [`BLOCK_HEADER_SIZE`].
    pub fn size(&self) -> usize {
        self.data.size
    }

    /// Total block size including [`BLOCK_HEADER_SIZE`].
    pub fn total_size(&self) -> usize {
        BLOCK_HEADER_SIZE + self.data.size
    }
}
