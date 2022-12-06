use std::{mem, ptr::NonNull};

use crate::{
    header::Header,
    list::{LinkedList, Node},
    region::Region,
};

/// Minimum block size in bytes. Read the documentation of the data structures
/// used for this allocator to understand why it has this value. Specifically,
/// see [`Header<T>`], [`Block`], [`Region`], [`LinkedList<T>`] and especially
/// [`FreeListNode`].
pub const MIN_BLOCK_SIZE: usize = mem::size_of::<FreeListNode>();

/// Block header size in bytes. See [`Header<T>`] and [`Block`].
pub const BLOCK_HEADER_SIZE: usize = mem::size_of::<Header<Block>>();

/// Memory block specific data. All headers are also linked list nodes, see
/// [`Header<T>`]. In this case, a complete block header would be
/// [`Node<Block>`], also known as [`Header<Block>`]. Here's a graphical
/// representation of how it looks like in memory:
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
#[derive(Clone, Copy)]
pub struct Block {
    /// Memory region where this block is located.
    pub region: NonNull<Header<Region>>,
    /// Size of the block excluding [`Header<Block>`] size.
    pub size: usize,
    /// Whether this block can be used or not.
    pub is_free: bool,
}

/// When a block is free we'll use the content of the block to store a free
/// list, that is, a linked list of _only_ free blocks. Since we want a doubly
/// linked list, we need to store 2 pointers, one for the previous block and
/// another one for the next free block. This is how a free block would look
/// like in memory:
///
/// ```text
/// +----------------------------+
/// | pointer to next block      | <--+
/// +----------------------------+    |
/// | pointer to prev block      |    |
/// +----------------------------+    | Node<Block> struct.
/// | rest of fields             |    |
/// +----------------------------+    |
/// |          ......            | <--+
/// +----------------------------+
/// | pointer to next free block | <--+
/// +----------------------------+    | Node<()> struct.
/// | pointer to prev free block | <--+
/// +----------------------------+
/// |     Rest of user data      | <--+
/// |          ......            |    | Rest of content. This could be 0 bytes.
/// |          ......            | <--+
/// +----------------------------+
/// ```
///
/// Free blocks could point to blocks located in different regions, since _all_
/// free blocks are linked. See this representation:
///
/// ```text
///                   Points to free block in next region       Points to same region
///                +--------------------------------------+   +-----------------------+
///                |                                      |   |                       |
/// +--------+-----|------------------+      +--------+---|---|-----------------------|-----+
/// |        | +---|---+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// +--------+------------------------+      ---------+-------------------------------------+
/// ```
///
/// However, there's a little catch. As you can see we use `Node<()>` to
/// represent a node of the free list. That's because, first of all, we want to
/// reuse the [`LinkedList<T>`] implementation. Secondly, there's no additional
/// metadata associated to free blocks other than pointers to previous and next
/// free blocks. All other data such as block size or region is located in the
/// `Node<Block>` struct right above `Node<()>`, as you can see in the memory
/// representation.
///
/// The [`Node<T>`] type stores the pointers we need plus some other data, so if
/// we give it a zero sized `T` we can reuse it for the free list without
/// declaring a new type or consuming additional memory. But this has some
/// implications over simply storing `*mut Node<Block>` in the free block
/// content space.
///
/// [`Node<T>`] can only point to instances of itself, so [`Node<()>`] can only
/// point to [`Node<()>`], otherwise [`LinkedList<T>`] implementation won't work
/// for this type. Therefore, the free list doesn't contain pointers to the
/// headers of free blocks, it contains pointers to the *content* of free
/// blocks. If we want to obtain the actual block header given a `Node<()>`, we
/// have to substract [`BLOCK_HEADER_SIZE`] to its address and cast to
/// [`Header<Block>`] or [`Node<Block>`].
///
/// It's not so intuitive because the free list should be [`LinkedList<Block>`],
/// but with this implementation it becomes [`LinkedList<()>`] instead. This,
/// however, allows us to reuse [`LinkedList<T>`] without having to implement
/// traits for different types of nodes to give us their next and previous or
/// having to rely on macros to generate code for that. The only incovinience
/// is that the free list points to content of free blocks instead of their
/// header, but we can easily obtain the actual header.
///
/// I suggest that you check out the following commit, which doesn't have
/// [`LinkedList<T>`], to understand why this method reduces boilerplate:
/// [`37b7752e2daa6707c93cd7badfa85c168f09aac8`](https://github.com/antoniosarosi/memalloc-rust/blob/37b7752e2daa6707c93cd7badfa85c168f09aac8/src/mmap.rs#L649-L687)
pub type FreeListNode = Node<()>;

/// See [`FreeListNode`].
pub type FreeList = LinkedList<()>;

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
    /// is `Header<Block>` all `Block` fields have to be accesses through
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

impl FreeList {
    /// Helper function for adding blocks to the free list. `block` must be
    /// valid.
    pub unsafe fn append_block(&mut self, mut block: NonNull<Header<Block>>) {
        self.append((), Header::content_address_of(block));
        block.as_mut().data.is_free = true;
    }

    /// Removes `block` from the free list. `block` must be valid.
    pub unsafe fn remove_block(&mut self, mut block: NonNull<Header<Block>>) {
        self.remove(NonNull::new_unchecked(
            Header::content_address_of(block).as_ptr() as *mut FreeListNode,
        ));
        block.as_mut().data.is_free = false;
    }

    /// Returns a reference to the block header of the first block in the free
    /// list. Not used internally, for now we only need it for testing.
    #[allow(dead_code)]
    pub unsafe fn first_free_block(&self) -> Option<&Header<Block>> {
        self.head.and_then(|node| {
            let block = Header::<Block>::from_free_list_node(node);
            Some(block.as_ref())
        })
    }
}
