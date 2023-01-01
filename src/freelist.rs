use std::ptr::NonNull;

use crate::{
    block::Block,
    header::Header,
    list::{LinkedList, Node},
};

/// See [`crate::block::Block`] and [`crate::region::Region`] first.
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
/// # Free blocks positioning
///
/// Free blocks may point to blocks located in different regions, since _all_
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
/// +--------+------------------------+      +--------+-------------------------------------+
/// ```
///
/// Also, note that the first free block doesn't have to be located in the
/// first region, and the last free block doesn't have to be located in the last
/// region either. Free blocks can be located anywhere. Consider this case:
///
/// 1. We get a new region from the kernel 4096 bytes in length.
///
/// 2. We create one single block in this region where we can allocate a maximum
/// of 4096 - R - B bytes, where R = [`crate::region::REGION_HEADER_SIZE`] and
/// B = [`crate::block::BLOCK_HEADER_SIZE`]. This would be the current state:
///
/// ```text
///
/// +--------+-------------------------------------+
/// |        | +---------------------------------+ |
/// | Region | |          Big Free Block         | |
/// |        | +---------------------------------+ |
/// +--------+-------------------------------------+
///                             ^
///                             |
///                             +--- Head of the free list points to this block.
///                                  Tail also points here (only one node).
/// ```
///
/// 3. The block takes up all the space, if a subsequent allocation is smaller
/// than the block size, then the block will be split in two different blocks
/// and the second block will become the first and only free block.
///
/// ```text
/// +--------+-------------------------------------+
/// |        | +-------+    +--------------------+ |
/// | Region | | Alloc | -> |        Free        | |
/// |        | +-------+    +--------------------+ |
/// +--------+-------------------------------------+
///                                    ^
///                                    |
///                                    +--- Head and tail of the free list.
/// ```
///
/// 4. If the user makes another allocation that is smaller than our free block,
/// the splitting algorithm does its job again:
///
/// ```text
/// +--------+-------------------------------------+
/// |        | +-------+    +-------+    +-------+ |
/// | Region | | Alloc | -> | Alloc | -> | Free  | |
/// |        | +-------+    +-------+    +-------+ |
/// +--------+-------------------------------------+
///                                          ^
///                                          |
///                                          +--- Head and tail.
/// ```
///
/// 5. Now the user deallocates the first block, se we have 2 free blocks:
///
/// ```text
///                +-------------------------+
///                |                         |
/// +--------+-----|-------------------------|-----+
/// |        | +---|---+    +-------+    +---|---+ |
/// | Region | | Free  | -> | Alloc | -> | Free  | |
/// |        | +-------+    +-------+    +-------+ |
/// +--------+-------------------------------------+
///                ^                         ^
///                |                         |
///                +--- Tail                 +--- Head
/// ```
///
/// At this point, if the user makes another allocation that doesn't fit any of
/// the two free blocks that we have we'll need to request another region, so
/// the current tail will point to the new tail located in the new region.
/// Because of this, we cannot make any assumptions regarding positioning when
/// it comes to free blocks. All this process of splitting blocks, merging them
/// again and updating the free list is handled at [`crate::bucket::Bucket`].
///
/// # Free list implementation
///
/// Now, going back to the inernals of the free list, there's a little catch. As
/// you can see we use [`Node<()>`] to represent a node of the free list. That's
/// because, first of all, we want to reuse the [`LinkedList<T>`]
/// implementation. Secondly, there's no additional metadata associated to free
/// blocks other than pointers to previous and next free blocks. All other data
/// such as block size or region is located in the [`Node<Block>`] struct right
/// above [`Node<()>`], as you can see in the memory representation.
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
/// blocks. If we want to obtain the actual block header given a [`Node<()>`],
/// we have to substract [`crate::block::BLOCK_HEADER_SIZE`] to its address and
/// cast to [`Header<Block>`] or [`Node<Block>`].
///
/// It's not so intuitive because the free list should be [`LinkedList<Block>`],
/// it's list of free blocks after all, so it should point to blocks themselves.
/// But with this implementation it becomes [`LinkedList<()>`] instead. This,
/// however, allows us to reuse [`LinkedList<T>`] without having to implement
/// traits for different types of nodes to give us their next and previous or
/// having to rely on macros to generate code for that. The only incovinience is
/// that the free list points to content of free blocks instead of their header,
/// but we can easily obtain the actual header.
///
/// Note that generally we never store pointers to the contents of a block
/// because the user also has pointers to those addresses, so we don't want
/// aliasing because that would probably cause issues with references. However,
/// if a block has been deallocated, we can actually point to its content
/// because the user doesn't have pointers to the content address anymore, all
/// of them should have been dropped. If users still maintain pointers to
/// deallocated addresses they will run into use after free, so they better not
/// use such pointers! [Miri](https://github.com/rust-lang/miri) is really
/// helpful for finding bugs related to this kind of problems.
pub(crate) type FreeListNode = Node<()>;

/// See [`FreeListNode`].
pub(crate) type FreeList = LinkedList<()>;

impl FreeList {
    /// Helper function for adding blocks to the free list. `block` must be
    /// valid.
    pub unsafe fn append_block(&mut self, mut block: NonNull<Header<Block>>) {
        self.append((), Header::content_address_of(block));
        block.as_mut().data.is_free = true;
    }

    /// Removes `block` from the free list. `block` must be valid.
    pub unsafe fn remove_block(&mut self, mut block: NonNull<Header<Block>>) {
        self.remove(Header::content_address_of(block).cast());
        block.as_mut().data.is_free = false;
    }

    /// Returns a reference to the block header of the first block in the free
    /// list. Not used internally, for now we only need it for testing.
    #[cfg(test)]
    pub unsafe fn first_free_block(&self) -> Option<&Header<Block>> {
        self.first().map(|node| {
            let block = Header::<Block>::from_free_list_node(node);
            block.as_ref()
        })
    }

    /// Free list nodes are a little bit harder to iterate because they don't
    /// point to block headers, so let's make it easier.
    pub unsafe fn iter_blocks(&self) -> impl Iterator<Item = NonNull<Header<Block>>> {
        self.iter()
            .map(|node| Header::<Block>::from_free_list_node(node))
    }
}
