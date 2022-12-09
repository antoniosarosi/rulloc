use std::{
    alloc::{AllocError, Layout},
    ptr::NonNull,
};

use crate::{
    alignment,
    block::{Block, BLOCK_HEADER_SIZE, MIN_BLOCK_SIZE},
    freelist::FreeList,
    header::Header,
    list::LinkedList,
    mmap::{mmap, munmap},
    region::{determine_region_length, Region, REGION_HEADER_SIZE},
    Pointer,
};

/// This, on itself, is actually a memory allocator. But we use multiple of
/// them for optimization purposes. Basically, we can configure different
/// buckets that will perform allocations of different sizes. So, for example,
/// we might have a bucket for small allocations, say 128 bytes or less, and
/// another bucket for allocations larger than 128 bytes. This method is called
/// segregation list or segregation buckets. There's a visual representation
/// at [`crate::allocator`]. The individual bucket on itself is only concerned
/// about regions and blocks. See [`Header`], [`Block`], [`Region`] and
/// [`FreeList`] for a full picture. To reiterate, this is what the bucket
/// stores:
///
/// ```text
///                              Next Free Block                    Next Free Block
///                  |------------------------------------+   +-----------------------+
///                  |                                    |   |                       |
/// +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
/// |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// +--------+------------------------+      ---------+-------------------------------------+
///     ^          ^                              ^                                   ^
///     |          |                              |                                   |
///     |          +--- free_blocks.head          |                                   +--- free_blocks.tail
///     |                                         |
///     +--- regions.head                         +--- regions.tail
///
/// ```
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bucket {
    /// Free list.
    free_blocks: FreeList,
    /// All regions mapped by this bucket.
    regions: LinkedList<Region>,
}

impl Bucket {
    /// Builds a new empty [`Bucket`].
    pub const fn new() -> Self {
        Self {
            free_blocks: FreeList::new(),
            regions: LinkedList::new(),
        }
    }

    /// Only used for testing at [`crate::allocator`].
    #[cfg(test)]
    pub fn regions(&self) -> &LinkedList<Region> {
        &self.regions
    }

    /// Allocates a new block that can fit at least `layout.size()` bytes.
    /// Because of alignment and headers, it might allocate a bigger block than
    /// needed. As long as no more than `layout.pad_to_align().size()` bytes are
    /// written on the content part of the block it should be fine.
    pub unsafe fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let size = alignment::minimum_block_size_needed_for(layout);

        let free_block = match self.find_free_block(size) {
            Some(block) => block,
            None => self.request_region(size)?.as_ref().first_block(),
        };

        self.split_free_block_if_possible(free_block, size);
        self.free_blocks.remove_block(free_block);

        let address = self.introduce_padding(free_block, layout.align());

        Ok(address)
    }

    /// Deallocates the given pointer. Memory might not be returned to the OS
    /// if the region where `address` is located still contains used blocks.
    /// However, the freed block will be reused later if possible.
    pub unsafe fn deallocate(&mut self, address: NonNull<u8>, layout: Layout) {
        let mut block = if layout.align() <= alignment::POINTER_SIZE {
            Header::<Block>::from_content_address(address)
        } else {
            Header::<Block>::from_aligned_address(address)
        };

        // This block is now free as it is about to be deallocated.
        self.free_blocks.append_block(block);

        // If previous block is merged then the address will change.
        block = self.merge_free_blocks_if_possible(block);

        let region = block.as_ref().data.region;

        // All blocks have been merged into one, so we can return this region
        // back to the kernel.
        if region.as_ref().num_blocks() == 1 {
            // The only block in this region is no longer valid because the
            // region is about to be unmapped.
            self.free_blocks.remove_block(block);

            // Region has to be removed before unmapping, otherwise seg fault.
            self.regions.remove(region);

            munmap(region.cast(), region.as_ref().total_size());
        }
    }

    /// This function performs the algorithm described at
    /// [`alignment::AlignmentBackPointer`]. The caller must guarantee that
    /// the given block meets the size constraints needed to introduce enough
    /// padding without overriding other headers. See
    /// [`alignment::minimum_block_size_needed_for`].
    unsafe fn introduce_padding(
        &self,
        block: NonNull<Header<Block>>,
        align: usize,
    ) -> NonNull<[u8]> {
        let content_address = Header::content_address_of(block);

        if align <= alignment::POINTER_SIZE {
            return NonNull::slice_from_raw_parts(content_address, block.as_ref().size());
        }

        let next_aligned = alignment::next_aligned(content_address, align);

        let back_pointer = alignment::back_pointer_of(next_aligned).as_ptr();
        *back_pointer = block;

        let padding = next_aligned.as_ptr().offset_from(content_address.as_ptr()) as usize;

        return NonNull::slice_from_raw_parts(next_aligned, block.as_ref().size() - padding);
    }

    /// Requests a new memory region from the kernel where we can fit `size`
    /// bytes plus headers. See [`determine_region_length`]. The returned region
    /// will be initialized with a free block that spans across the entire
    /// region.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes (must be aligned to word size) that
    /// need to be allocated **without including any headers**.
    unsafe fn request_region(
        &mut self,
        size: usize,
    ) -> Result<NonNull<Header<Region>>, AllocError> {
        let length = determine_region_length(size);

        let Some(address) = mmap(length) else {
            return Err(AllocError);
        };

        let mut region = self.regions.append(
            Region {
                blocks: LinkedList::new(),
                size: length - REGION_HEADER_SIZE,
            },
            address,
        );

        let block = region.as_mut().data.blocks.append(
            Block {
                size: region.as_ref().size() - BLOCK_HEADER_SIZE,
                is_free: true,
                region,
            },
            Header::content_address_of(region),
        );

        self.free_blocks.append_block(block);

        Ok(region)
    }

    /// Returns the first free block in the free list or `None` if we didn't
    /// find any.
    unsafe fn find_free_block(&self, size: usize) -> Pointer<Header<Block>> {
        let mut current = self.free_blocks.first();

        while let Some(node) = current {
            let block = Header::<Block>::from_free_list_node(node);

            if block.as_ref().size() >= size {
                return Some(block);
            }

            current = node.as_ref().next;
        }

        None
    }

    /// Block splitting algorithm implementation. Let's say we have a free block
    /// that can hold 64 bytes and a request to allocate 8 bytes has been made.
    /// We'll split the free block in two different blocks, like so:
    ///
    /// **Before**:
    ///
    /// ```text
    ///         +-->  +-----------+
    ///         |     |   Header  | <- H bytes (depends on word size and stuff).
    /// Block   |     +-----------+
    ///         |     |  Content  | <- 64 bytes.
    ///         +-->  +-----------+
    /// ```
    /// **After**:
    ///
    /// ```text
    ///         +-->  +-----------+
    ///         |     |   Header  | <- H bytes.
    /// Block 1 |     +-----------+
    ///         |     |  Content  | <- 8 bytes.
    ///         +-->  +-----------+
    ///         |     |   Header  | <- H bytes.
    /// Block 2 |     +-----------+
    ///         |     |  Content  | <- 64 bytes - 8 bytes - H bytes.
    ///         +-->  +-----------+
    /// ```
    unsafe fn split_free_block_if_possible(
        &mut self,
        mut block: NonNull<Header<Block>>,
        size: usize,
    ) {
        // If there's not enough space available we can't split the block.
        if block.as_ref().size() < size + BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE {
            return;
        }

        // If we can, the block is located `size` bytes after the initial
        // content address of the current block.
        let address = Header::content_address_of(block).as_ptr().add(size);
        let mut region = block.as_ref().data.region;

        // Append new block next to current block.
        let new_block = region.as_mut().data.blocks.insert_after(
            block,
            Block {
                size: block.as_ref().data.size - size - BLOCK_HEADER_SIZE,
                is_free: true,
                region,
            },
            NonNull::new_unchecked(address),
        );

        self.free_blocks.append_block(new_block);

        // The current block can only hold `size` bytes from now on.
        block.as_mut().data.size = size;
    }

    /// This function performs the inverse of [`Self::split_free_block_if_possible`].
    /// If surrounding blocks are free, then we'll merge them all into one
    /// bigger block. This is called block coalescing or block merging.
    ///
    /// **Before**:
    ///
    /// ```text
    ///                         +-->  +-----------+
    ///                         |     |   Header  | <- H bytes.
    /// Block A, Free           |     +-----------+
    ///                         |     |  Content  | <- A bytes.
    ///                         +-->  +-----------+
    ///                         |     |   Header  | <- H bytes.
    /// Block B, Recently freed |     +-----------+
    ///                         |     |  Content  | <- B bytes.
    ///                         +-->  +-----------+
    ///                         |     |   Header  | <- H bytes.
    /// Block C, Free           |     +-----------+
    ///                         |     |  Content  | <- C bytes.
    ///                         +-->  +-----------+
    /// ```
    ///
    /// **After**:
    ///
    /// ```text
    ///
    ///                         +-->  +-----------+
    ///                         |     |   Header  | <- H bytes.
    /// Block D, Bigger block   |     +-----------+
    ///                         |     |  Content  | <- A + B + C + 2H bytes.
    ///                         +-->  +-----------+
    /// ```
    ///
    /// Note that only one of the surrounding is merged if the other one is not
    /// free. Also, if the previous block is merged, then the address of the
    /// current block changes. That's why we have to return a pointer to a
    /// block.
    #[rustfmt::skip]
    unsafe fn merge_free_blocks_if_possible(
        &mut self,
        mut block: NonNull<Header<Block>>,
    ) -> NonNull<Header<Block>> {
        if block.as_ref().next.is_some_and(|next| next.as_ref().is_free()) {
            self.merge_next_block(block);
        }

        if block.as_ref().prev.is_some_and(|prev| prev.as_ref().is_free()) {
            block = block.as_ref().prev.unwrap();
            self.merge_next_block(block);
        }

        block
    }

    /// Helper for block merging algorithm. Blocks can only be merged from right
    /// to left, or from next to current, because the new bigger block address
    /// should be the address of the first free block. So, given a valid block,
    /// this function will merge the next adjacent block into the given block
    /// and update all data structures. This functions also assumes that the
    /// given block is free and the next block to it is also free and not null.
    ///
    /// ```text
    /// +----------------+---------------+
    /// |    Block A     |   Block B     |
    /// +----------------+---------------+
    ///        ^                 |
    ///        |                 |
    ///        +-----------------+
    ///           Merge B into A
    /// ```
    unsafe fn merge_next_block(&mut self, mut block: NonNull<Header<Block>>) {
        let next = block.as_ref().next.unwrap();

        // First update free list. The new bigger block will become the
        // last block, and the 2 old smaller blocks will "dissapear" from the
        // list.
        self.free_blocks.remove_block(next);
        self.free_blocks.remove_block(block);
        self.free_blocks.append_block(block);

        // Now this block is bigger.
        block.as_mut().data.size += next.as_ref().total_size();

        // Next block doesn't exist anymore.
        block.as_mut().region_mut().data.blocks.remove(next);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::PAGE_SIZE;

    #[test]
    fn regions_and_blocks() {
        unsafe {
            let mut bucket = Bucket::new();

            // Request 1 byte, should call `mmap` with length of PAGE_SIZE.
            let first_layout = Layout::new::<u8>();
            let first_addr = bucket.allocate(first_layout).unwrap().cast::<u8>();

            // We'll use this later to check memory corruption.
            *first_addr.as_ptr() = 69;

            // First region should be PAGE_SIZE in length.
            let first_region = bucket.regions.first().unwrap();
            assert_eq!(first_region.as_ref().total_size(), PAGE_SIZE);
            assert_eq!(bucket.regions.len(), 1);

            let first_block = first_region.as_ref().first_block();

            // First block should be located after the region header.
            assert_eq!(
                first_block.as_ptr() as usize - first_region.as_ptr() as usize,
                REGION_HEADER_SIZE
            );

            // Size of first block should be minimum size because we've
            // requested 1 byte only.
            assert_eq!(first_block.as_ref().size(), MIN_BLOCK_SIZE);
            assert_eq!(bucket.free_blocks.len(), 1);

            // Region should have two blocks, we are using the first one and the
            // other one is free.
            assert_eq!(first_region.as_ref().num_blocks(), 2);

            // First free block should match this size because there are
            // two blocks in the region and one of them has minimum size.
            assert_eq!(
                bucket.free_blocks.first_free_block().unwrap().size(),
                PAGE_SIZE - REGION_HEADER_SIZE - BLOCK_HEADER_SIZE * 2 - MIN_BLOCK_SIZE
            );

            // The remaining free block should be split in two when allocating
            // less size than it can hold.
            let second_layout = Layout::array::<u8>(PAGE_SIZE / 2).unwrap();
            let second_addr = bucket.allocate(second_layout).unwrap().cast::<u8>();

            // We'll check corruption later.
            for _ in 0..second_layout.size() {
                *second_addr.as_ptr() = 69;
            }

            // There are 3 blocks now, last one is still free.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 1);

            // Lets try to allocate the entire remaining free block.
            let remaining_size = PAGE_SIZE
                - REGION_HEADER_SIZE
                - (BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE) // First Alloc
                - (BLOCK_HEADER_SIZE + PAGE_SIZE / 2) // Second Alloc
                - BLOCK_HEADER_SIZE;
            let third_layout = Layout::array::<u8>(remaining_size).unwrap();
            let third_addr = bucket.allocate(third_layout).unwrap().cast::<u8>();

            for _ in 0..third_layout.size() {
                *third_addr.as_ptr() = 69;
            }

            // Number of blocks hasn't changed, but we don't have free blocks
            // anymore.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 0);

            // Time for checking memory corruption
            assert_eq!(*first_addr.as_ptr(), 69);
            for _ in 0..second_layout.size() {
                assert_eq!(*second_addr.as_ptr(), 69);
            }
            for _ in 0..third_layout.size() {
                assert_eq!(*third_addr.as_ptr(), 69);
            }

            // Let's request a bigger chunk so that a new region is used.
            let fourth_layout = Layout::array::<u8>(PAGE_SIZE * 2 - PAGE_SIZE / 2).unwrap();
            let fourth_addr = bucket.allocate(fourth_layout).unwrap().cast::<u8>();

            for _ in 0..fourth_layout.size() {
                *fourth_addr.as_ptr() = 69;
            }

            // We should have a new region and a new free block now.
            assert_eq!(bucket.regions.len(), 2);
            assert_eq!(bucket.free_blocks.len(), 1);

            // Let's play with dealloc.
            bucket.deallocate(first_addr, first_layout);

            // After deallocating the first block, we should have a new free
            // block but the number of blocks in the region shouldn't change
            // because no coalescing can happen.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 2);

            bucket.deallocate(third_addr, third_layout);

            // Again, after deallocating the third block we should have a new
            // free block but the number of block in the region doesn't change.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 3);

            // Now here comes the magic, if we deallocate second addr all blocks
            // in region one should be merged and region should be returned to
            // the kernel.
            bucket.deallocate(second_addr, second_layout);
            assert_eq!(bucket.regions.len(), 1);
            assert_eq!(bucket.free_blocks.len(), 1);

            // Check mem corruption in the last block
            for _ in 0..fourth_layout.size() {
                assert_eq!(*fourth_addr.as_ptr(), 69);
            }

            // Deallocating fourh address should unmap the last region.
            bucket.deallocate(fourth_addr, fourth_layout);
            assert_eq!(bucket.regions.len(), 0);
            assert_eq!(bucket.free_blocks.len(), 0);
        }
    }
}
