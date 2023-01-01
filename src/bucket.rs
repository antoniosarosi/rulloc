use std::{
    alloc::{AllocError, Layout},
    mem::ManuallyDrop,
    ptr::{self, NonNull},
};

use crate::{
    alignment,
    block::{Block, BLOCK_HEADER_SIZE, MIN_BLOCK_SIZE},
    freelist::FreeList,
    header::Header,
    list::LinkedList,
    platform,
    realloc::{Realloc, ReallocMethod},
    region::{determine_region_length, Region, REGION_HEADER_SIZE},
    AllocResult, Pointer,
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
///                  +------------------------------------+   +-----------------------+
///                  |                                    |   |                       |
/// +--------+-------|----------------+      +--------+---|---|-----------------------|-----+
/// |        | +-----|-+    +-------+ |      |        | +-|---|-+    +-------+    +---|---+ |
/// | Region | | Free  | -> | Block | | ---> | Region | | Free  | -> | Block | -> | Free  | |
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// +--------+------------------------+      +--------+-------------------------------------+
///     ^          ^                              ^                                   ^
///     |          |                              |                                   |
///     |          +--- free_blocks.head          |                                   +--- free_blocks.tail
///     |                                         |
///     +--- regions.head                         +--- regions.tail
///
/// ```
pub(crate) struct Bucket {
    /// Free list.
    free_blocks: ManuallyDrop<FreeList>,
    /// All regions mapped by this bucket.
    regions: ManuallyDrop<LinkedList<Region>>,
}

impl Bucket {
    /// Builds a new empty [`Bucket`].
    pub const fn new() -> Self {
        Self {
            free_blocks: ManuallyDrop::new(FreeList::new()),
            regions: ManuallyDrop::new(LinkedList::new()),
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
    pub unsafe fn allocate(&mut self, layout: Layout) -> AllocResult {
        let size = alignment::minimum_block_size_needed_for(layout);

        let free_block = match self.find_free_block(size) {
            Some(block) => block,
            None => self.request_region(size)?.as_ref().first_block(),
        };

        self.split_block_if_possible(free_block, size);
        self.free_blocks.remove_block(free_block);

        let address = self.add_padding_if_needed(free_block, layout.align());

        Ok(address)
    }

    /// Deallocates the given pointer. Memory might not be returned to the OS
    /// if the region where `address` is located still contains used blocks.
    /// However, the freed block will be reused later if possible.
    pub unsafe fn deallocate(&mut self, address: NonNull<u8>, layout: Layout) {
        let mut block = Header::<Block>::from_allocated_pointer(address, layout);

        // This block is now free as it is about to be deallocated.
        self.free_blocks.append_block(block);

        // If previous block is merged then the address will change.
        block = self.merge_surrounding_free_blocks_if_possible(block);

        let region = block.as_ref().data.region;

        // All blocks have been merged into one, so we can return this region
        // back to the kernel.
        if region.as_ref().num_blocks() == 1 {
            // The only block in this region is no longer valid because the
            // region is about to be unmapped.
            self.free_blocks.remove_block(block);

            // Region has to be removed before unmapping, otherwise seg fault.
            self.regions.remove(region);

            platform::return_memory(region.cast(), region.as_ref().total_size());
        }
    }

    /// Executes the reallocation specified by `realloc`. When possible,
    /// reallocation is done in place to avoid copying contents from one block
    /// to another, but changes in alignment constraints might prevent that.
    pub unsafe fn reallocate(&mut self, realloc: &Realloc) -> AllocResult {
        // Reallocation is more complicated than allocation or deallocation,
        // so study the code in the functions below to understand what's
        // happening.
        self.try_reallocate_in_place(realloc)
            .or_else(|_| self.try_reallocate_on_another_block(realloc))
    }

    /// Returns the first free block in the free list or `None` if we didn't
    /// find any.
    unsafe fn find_free_block(&self, size: usize) -> Pointer<Header<Block>> {
        self.free_blocks
            .iter_blocks()
            .find(|block| block.as_ref().size() >= size)
    }

    /// This function executes the algorithm described at
    /// [`alignment::AlignmentBackPointer`]. The caller must guarantee that
    /// the given block meets the size constraints needed to introduce enough
    /// padding without overriding other headers. See
    /// [`alignment::minimum_block_size_needed_for`].
    unsafe fn add_padding_if_needed(
        &self,
        block: NonNull<Header<Block>>,
        align: usize,
    ) -> NonNull<[u8]> {
        let content_address = Header::content_address_of(block);

        if align <= alignment::POINTER_SIZE {
            return NonNull::slice_from_raw_parts(content_address, block.as_ref().size());
        }

        let (next_aligned, padding) = alignment::next_aligned(content_address, align);

        ptr::write(alignment::back_pointer_of(next_aligned).as_ptr(), block);

        return NonNull::slice_from_raw_parts(next_aligned, block.as_ref().size() - padding);
    }

    /// Requests a new memory region from the kernel where we can fit `size`
    /// bytes plus headers. See [`determine_region_length`]. The returned region
    /// will be already initialized with one single free block that takes up all
    /// the space minus headers:
    ///
    /// ```text
    /// +--------+-------------------------------------+
    /// |        | +---------------------------------+ |
    /// | Region | |          Big Free Block         | |
    /// |        | +---------------------------------+ |
    /// +--------+-------------------------------------+
    /// ```
    ///
    /// The block must be split if it is too large in order to reduce
    /// fragmentation.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes (must be aligned to power of 2) that
    /// need to be allocated **without including any headers**.
    unsafe fn request_region(
        &mut self,
        size: usize,
    ) -> Result<NonNull<Header<Region>>, AllocError> {
        let length = determine_region_length(size);

        let address = platform::request_memory(length).ok_or(AllocError)?;

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

    /// Block splitting algorithm implementation. Let's say we have a free block
    /// that can hold 128 bytes and a request to allocate 24 bytes has been
    /// made. We'll split the free block in two different blocks, like so:
    ///
    /// **Before**:
    ///
    /// ```text
    ///         +-->  +-----------+
    ///         |     |   Header  | <- H bytes (depends on word size and stuff).
    /// Block   |     +-----------+
    ///         |     |  Content  | <- 128 bytes.
    ///         +-->  +-----------+
    /// ```
    /// **After**:
    ///
    /// ```text
    ///         +-->  +-----------+
    ///         |     |   Header  | <- H bytes.
    /// Block 1 |     +-----------+
    ///         |     |  Content  | <- 24 bytes.
    ///         +-->  +-----------+
    ///         |     |   Header  | <- H bytes.
    /// Block 2 |     +-----------+
    ///         |     |  Content  | <- 128 bytes - 24 bytes - H bytes.
    ///         +-->  +-----------+
    /// ```
    ///
    /// The block doesn't necessarily have to be free, it might be in use but
    /// we want to shrink it. See [`Self::shrink_block`]. This function does not
    /// touch the contents of the block, it only changes it's header to reflect
    /// the new size. On the other hand, the new block created in the splitting
    /// process is automatically added to the free list.
    ///
    /// # Safety
    ///
    /// User content in this block can never exceed `size` bytes, this is only
    /// relevant for shrinking used blocks. If padding was introduced in this
    /// block to meet alignment constraints, the caller must guarantee that
    /// padding is included in `size`.
    unsafe fn split_block_if_possible(&mut self, mut block: NonNull<Header<Block>>, size: usize) {
        // If there's not enough space available we can't split the block.
        if block.as_ref().size() < size + BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE {
            return;
        }

        // If we can, the new block will be located `size` bytes after the
        // initial content address of the current block.
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

    /// This function performs the inverse of [`Self::split_block_if_possible`].
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
    /// Note that only one of the surrounding blocks is merged if the other one
    /// is not free. Also, if the previous block is merged, then the address of
    /// the current block changes. That's why we have to return a pointer to a
    /// block.
    ///
    /// # Safety
    ///
    /// Unlike [`Self::split_block_if_possible`], the caller must guarantee that
    /// `block` is free in this case.
    #[rustfmt::skip]
    unsafe fn merge_surrounding_free_blocks_if_possible(
        &mut self,
        mut block: NonNull<Header<Block>>,
    ) -> NonNull<Header<Block>> {
        if block.as_ref().next.is_some_and(|next| next.as_ref().is_free()) {
            self.merge_next_adjacent_free_block(block);
        }

        if block.as_ref().prev.is_some_and(|prev| prev.as_ref().is_free()) {
            block = block.as_ref().prev.unwrap();
            self.merge_next_adjacent_free_block(block);
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
    unsafe fn merge_next_adjacent_free_block(&mut self, block: NonNull<Header<Block>>) {
        let next = block.as_ref().next.unwrap();

        // First update free list. The new bigger block will become the
        // last block, and the 2 old smaller blocks will "dissapear" from the
        // list.
        self.free_blocks.remove_block(next);
        self.free_blocks.remove_block(block);
        self.free_blocks.append_block(block);

        self.expand_block_by_consuming_next(block);
    }

    /// This function expands `block` by consuming the block right next to it,
    /// without modifying the free list.
    ///
    /// # Panics
    ///
    /// Panics if there is no adjacent block next to the given block.
    unsafe fn expand_block_by_consuming_next(&mut self, mut block: NonNull<Header<Block>>) {
        let next = block.as_ref().next.unwrap();
        // Now this block is bigger.
        block.as_mut().data.size += next.as_ref().total_size();
        // Next block doesn't exist anymore.
        block.as_mut().region_mut().data.blocks.remove(next);
    }

    /// Optimized in place shrinking. The block can be split in two different
    /// blocks, creating a free block next to it, and if there was another free
    /// block next to the original block then they can be merged into one.
    ///
    /// Before:
    ///
    /// ```text
    /// +------------------------------------+    +------------+
    /// |    Used block about to be shrunk   | -> | Free Block |
    /// +------------------------------------+    +------------+
    /// ```
    ///
    /// After splitting:
    ///
    /// ```text
    /// +--------------+    +----------------+    +------------+
    /// | Shrunk Block | -> | New Free block | -> | Free Block |
    /// +--------------+    +----------------+    +------------+
    /// ```
    ///
    /// After merging:
    ///
    /// ```text
    /// +--------------+    +----------------------------------+
    /// | Shrunk Block | -> |   Resulting bigger free block    |
    /// +--------------+    +----------------------------------+
    /// ```
    unsafe fn shrink_block(&mut self, block: NonNull<Header<Block>>, new_size: usize) {
        self.split_block_if_possible(block, new_size);
        if let Some(next) = block.as_ref().next {
            self.merge_surrounding_free_blocks_if_possible(next);
        }
    }

    /// We will prioritize in place reallocations, this will only fail if the
    /// new size doesn't fit even after merging surrounding blocks or if
    /// alignment has increased drastically and we didn't find any aligned
    /// address in the current block.
    unsafe fn try_reallocate_in_place(&mut self, realloc: &Realloc) -> AllocResult {
        match realloc.method {
            ReallocMethod::Shrink => self.try_shrink_in_place(realloc),
            ReallocMethod::Grow => self.try_grow_in_place(realloc),
        }
    }

    /// If everything else fails, just find or create a new block and move
    /// the contents there.
    unsafe fn try_reallocate_on_another_block(&mut self, realloc: &Realloc) -> AllocResult {
        let new_address = self.allocate(realloc.new_layout)?;
        ptr::copy_nonoverlapping(
            realloc.address.as_ptr(),
            new_address.as_mut_ptr(),
            realloc.count(),
        );
        self.deallocate(realloc.address, realloc.old_layout);

        Ok(new_address)
    }

    /// If possible, this function will attempt to reallocate in place without
    /// merging adjacent blocks or moving the contents to a new block. This can
    /// be done especially when alignment changes, because that makes the
    /// padding change as well.
    unsafe fn try_reallocate_on_same_block(&mut self, realloc: &Realloc) -> AllocResult {
        // First let's try to see if we can fit the new layout in this block.
        let new_size = alignment::minimum_block_size_excluding_padding(realloc.new_layout);
        let (next_aligned, padding) = alignment::next_aligned(
            Header::<Block>::content_address_of(realloc.block),
            realloc.new_layout.align(),
        );

        // No luck, we can't use this block.
        if padding + new_size > realloc.block.as_ref().size() {
            return Err(AllocError);
        }

        // Now we know for sure that this block can be reused. The code below
        // deals with many edge cases even though it doesn't look like it does,
        // so let's break it down.
        //
        // We know that `alignment::next_aligned` function will return the first
        // aligned address after the content address of the block, and we also
        // know that there's enough space for padding because we've checked
        // that in the if statement above. So whatever the new alignment is, we
        // don't care, there's space for it.
        //
        // The `next_aligned` address might be located before the current
        // address (if alignment has decreased) or after the current address
        // (if alignment has increased), so we'll need to move the contents if
        // the address wasn't already aligned.
        //
        // But now here comes the magic: This code also works for `POINTER_SIZE`
        // alignment because `alignment::next_aligned` function will return the
        // content address of the block whenever the alignment is less than or
        // equal to `POINTER_SIZE`. So if the previous alignment was greater
        // than `POINTER_SIZE` we'll get rid of all the padding and reduce
        // fragmentation. If the previous alignment was already `POINTER_SIZE`
        // then the if statement below won't even run because `next_aligned` is
        // already equal to the given address!
        //
        // The only thing we have to do is write the back pointer if we added
        // padding, otherwise there's no back pointer. So this covers all the
        // combinatorics of `old_layout` and `new_layout`, both in terms of
        // alignment and size.
        //
        // Of course, this conclusion was reached after writing dozens of
        // if-else statements and noticing that the same code is repeated
        // everywhere. So this is just like simplifying a huge equation!
        if next_aligned != realloc.address {
            ptr::copy(
                realloc.address.as_ptr(),
                next_aligned.as_ptr(),
                realloc.count(),
            );
            if padding > 0 {
                ptr::write(
                    alignment::back_pointer_of(next_aligned).as_ptr(),
                    realloc.block,
                );
            }
        }

        // If we removed padding or size has decreased, maybe we can create
        // new free blocks next to this one.
        self.shrink_block(realloc.block, new_size + padding);

        Ok(NonNull::slice_from_raw_parts(
            next_aligned,
            realloc.block.as_ref().size() - padding,
        ))
    }

    /// Attempts to fit the layout of the new allocation using the surrounding
    /// free blocks. An error is returned if it's not possible.
    #[inline]
    unsafe fn try_grow_in_place(&mut self, realloc: &Realloc) -> AllocResult {
        self.try_reallocate_on_same_block(realloc)
            .or_else(|_| self.try_grow_by_merging_next_block(realloc))
            .or_else(|_| self.try_grow_by_merging_prev_block(realloc))
            .or_else(|_| self.try_grow_by_merging_both_blocks(realloc))
    }

    /// For symmetry with [`Self::try_grow_in_place`].
    #[inline]
    unsafe fn try_shrink_in_place(&mut self, realloc: &Realloc) -> AllocResult {
        self.try_reallocate_on_same_block(realloc)
    }

    /// As the name suggests, this function attempts to merge the next adjacent
    /// block to create a bigger one that fits the new layout.
    ///
    /// Before:
    ///
    /// ```text
    /// +------------------------------------+    +------------+
    /// |    Used block that needs to grow   | -> | Free Block |
    /// +------------------------------------+    +------------+
    /// ```
    ///
    /// After:
    ///
    /// ```text
    /// +------------------------------------------------------+
    /// |      Resulting block that fits the new layout        |
    /// +------------------------------------------------------+
    /// ```
    unsafe fn try_grow_by_merging_next_block(&mut self, realloc: &Realloc) -> AllocResult {
        let Realloc { block, .. } = realloc;
        let next = block.as_ref().next.ok_or(AllocError)?;
        self.try_grow_by_merging(&[*block, next], realloc)
    }

    /// Same as [`Self::try_grow_by_merging_next_block`] but using the previous
    /// block.
    ///
    /// Before:
    ///
    /// ```text
    /// +------------+    +------------------------------------+
    /// | Free Block | -> |    Used block that needs to grow   |
    /// +------------+    +------------------------------------+
    /// ```
    ///
    /// After:
    ///
    /// ```text
    /// +------------------------------------------------------+
    /// |      Resulting block that fits the new layout        |
    /// +------------------------------------------------------+
    /// ```
    unsafe fn try_grow_by_merging_prev_block(&mut self, realloc: &Realloc) -> AllocResult {
        let Realloc { block, .. } = realloc;
        let prev = block.as_ref().prev.ok_or(AllocError)?;
        self.try_grow_by_merging(&[prev, *block], realloc)
    }

    /// If we can't get a big enough block by merging the next or previous, how
    /// about merging both of them?
    ///
    /// Before:
    ///
    /// ```text
    /// +------------+    +----------------------------------+    +------------+
    /// | Free Block | -> |  Used block that needs to grow   | -> | Free Block |
    /// +------------+    +----------------------------------+    +------------+
    /// ```
    ///
    /// After:
    ///
    /// ```text
    /// +----------------------------------------------------------------------+
    /// |              Resulting block that fits the new layout                |
    /// +----------------------------------------------------------------------+
    /// ```
    unsafe fn try_grow_by_merging_both_blocks(&mut self, realloc: &Realloc) -> AllocResult {
        let Realloc { block, .. } = realloc;
        let next = block.as_ref().next.ok_or(AllocError)?;
        let prev = block.as_ref().prev.ok_or(AllocError)?;
        self.try_grow_by_merging(&[prev, *block, next], realloc)
    }

    /// Automation for merging blocks. `blocks` must contain only adjacent
    /// blocks, otherwise this will result in a disaster! See
    /// [`Self::try_grow_in_place`], [`Self::try_grow_by_merging_next_block`],
    /// [`Self::try_grow_by_merging_prev_block`] and
    /// [`Self::try_grow_by_merging_both_blocks`].
    unsafe fn try_grow_by_merging(
        &mut self,
        blocks: &[NonNull<Header<Block>>],
        realloc: &Realloc,
    ) -> AllocResult {
        let (starting_block, rest) = blocks.split_first().unwrap();

        let new_size = alignment::minimum_block_size_excluding_padding(realloc.new_layout);
        let padding = alignment::padding_needed_to_align(
            Header::<Block>::content_address_of(*starting_block),
            realloc.new_layout.align(),
        );

        let total_size = rest
            .iter()
            .fold(starting_block.as_ref().size(), |total, next| {
                total + next.as_ref().total_size()
            });

        if new_size + padding > total_size {
            return Err(AllocError);
        }

        for block in blocks.iter().rev().skip(1) {
            // We skipped one, there should always be a next block.
            let next = block.as_ref().next.unwrap();
            if next.as_ref().is_free() {
                self.free_blocks.remove_block(next);
            }
            if block.as_ref().is_free() {
                self.free_blocks.remove_block(*block);
            }
            self.expand_block_by_consuming_next(*block);
        }

        self.try_reallocate_on_same_block(&realloc.map(*starting_block))
    }
}

impl Drop for Bucket {
    fn drop(&mut self) {
        self.regions.iter().for_each(|region| unsafe {
            platform::return_memory(region.cast(), region.as_ref().total_size());
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alignment::AlignmentBackPointer,
        platform::{page_size, PAGE_SIZE},
    };

    fn check_mem_corruption(chunk: &[u8], corruption_check: u8) {
        for value in chunk {
            assert_eq!(value, &corruption_check);
        }
    }

    #[test]
    fn allocs_and_deallocs() {
        unsafe {
            let mut bucket = Bucket::new();

            // Request 1 byte, should call `mmap` with length of PAGE_SIZE.
            let first_layout = Layout::new::<u8>();
            let mut first_addr = bucket.allocate(first_layout).unwrap();

            // We'll use this later to check memory corruption. The allocator
            // should not touch the content of any block.
            let first_addr_corruption_check = 69;
            first_addr.as_mut().fill(first_addr_corruption_check);

            // First region should be PAGE_SIZE in length.
            let first_region = bucket.regions.first().unwrap();
            assert_eq!(first_region.as_ref().total_size(), PAGE_SIZE);
            assert_eq!(bucket.regions.len(), 1);

            // First block should be located after the region header.
            let first_block = first_region.as_ref().first_block();
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
            let mut second_addr = bucket.allocate(second_layout).unwrap();

            // We'll check corruption later.
            let second_addr_corruption_check = 42;
            second_addr.as_mut().fill(second_addr_corruption_check);

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
            let mut third_addr = bucket.allocate(third_layout).unwrap();

            let third_addr_corruption_check = 107;
            third_addr.as_mut().fill(third_addr_corruption_check);

            // Number of blocks hasn't changed, but we don't have free blocks
            // anymore.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 0);

            // Time for checking memory corruption
            check_mem_corruption(first_addr.as_ref(), first_addr_corruption_check);
            check_mem_corruption(second_addr.as_ref(), second_addr_corruption_check);
            check_mem_corruption(third_addr.as_ref(), third_addr_corruption_check);

            // Let's request a bigger chunk so that a new region is used.
            let fourth_layout = Layout::array::<u8>(PAGE_SIZE * 2 - PAGE_SIZE / 2).unwrap();
            let mut fourth_addr = bucket.allocate(fourth_layout).unwrap();

            let fourth_addr_corruption_check = 205;
            fourth_addr.as_mut().fill(fourth_addr_corruption_check);

            // We should have a new region and a new free block now.
            assert_eq!(bucket.regions.len(), 2);
            assert_eq!(bucket.free_blocks.len(), 1);

            // Let's play with dealloc.
            bucket.deallocate(first_addr.cast(), first_layout);

            // After deallocating the first block, we should have a new free
            // block but the number of blocks in the region shouldn't change
            // because no coalescing can happen.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 2);

            bucket.deallocate(third_addr.cast(), third_layout);

            // Again, after deallocating the third block we should have a new
            // free block but the number of block in the region doesn't change.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 3);

            // Now here comes the magic, if we deallocate second addr all blocks
            // in region one should be merged and region should be returned to
            // the kernel.
            bucket.deallocate(second_addr.cast(), second_layout);
            assert_eq!(bucket.regions.len(), 1);
            assert_eq!(bucket.free_blocks.len(), 1);

            // Check mem corruption in the last block
            check_mem_corruption(fourth_addr.as_ref(), fourth_addr_corruption_check);

            // Deallocating fourh address should unmap the last region.
            bucket.deallocate(fourth_addr.cast(), fourth_layout);
            assert_eq!(bucket.regions.len(), 0);
            assert_eq!(bucket.free_blocks.len(), 0);
        }
    }

    unsafe fn allocate_aligned(
        bucket: &mut Bucket,
        size: usize,
        align: usize,
        corruption_check: u8,
    ) -> (NonNull<u8>, Layout) {
        let layout = Layout::from_size_align(size, align).unwrap();
        let mut addr = bucket.allocate(layout).unwrap();

        // We are not actually performing aligned memory accesses,
        // but it doesn't matter, we just wanna check that we can
        // write to the requested memory and we don't seg fault.
        // We're not writing the entire layout when using Miri
        // because it's too slow, we'll just write to the addresses
        // that might cause problems, and Miri will catch bugs or
        // undefined behaviour.
        if cfg!(miri) {
            addr.as_mut()[0] = corruption_check;
            addr.as_mut()[size / 2] = corruption_check;
            addr.as_mut()[size - 1] = corruption_check;
        } else {
            addr.as_mut().fill(corruption_check);
        }

        assert_eq!(addr.as_mut_ptr() as usize % align, 0);

        (addr.cast(), layout)
    }

    unsafe fn deallocate_aligned(
        bucket: &mut Bucket,
        aligned_alloc: (NonNull<u8>, Layout),
        corruption_check: u8,
    ) {
        let (addr, layout) = aligned_alloc;
        let slice = NonNull::slice_from_raw_parts(addr, layout.size());
        if cfg!(miri) {
            check_mem_corruption(
                &[
                    slice.as_ref()[0],
                    slice.as_ref()[layout.size() / 2],
                    slice.as_ref()[layout.size() - 1],
                ],
                corruption_check,
            )
        } else {
            check_mem_corruption(slice.as_ref(), corruption_check);
        }
        bucket.deallocate(addr, layout);
    }

    #[test]
    fn strictly_aligned_allocs_and_deallocs() {
        unsafe {
            let mut bucket = Bucket::new();

            let layout = Layout::from_size_align(1, 16).unwrap();
            let address = bucket.allocate(layout).unwrap().cast::<u8>();

            assert_eq!(address.as_ptr() as usize % 16, 0);

            // Basic back pointer check
            let first_block = bucket.regions.first().unwrap().as_ref().first_block();
            let back_ptr = address.cast::<AlignmentBackPointer>().as_ptr().offset(-1);
            assert_eq!(*back_ptr, first_block);

            bucket.deallocate(address, layout);

            let alignments = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
            let sizes = [1, 2, 4, 8, 10, 20, 512, 1000, 2048, 4096];
            let mut allocations = [(NonNull::dangling(), Layout::new::<u8>()); 10];
            let corruption_check = 69;

            // Multiple allocations of same alignment.
            for align in alignments {
                for (i, size) in sizes.iter().enumerate() {
                    allocations[i] = allocate_aligned(&mut bucket, *size, align, corruption_check);
                }
                for allocation in allocations {
                    deallocate_aligned(&mut bucket, allocation, corruption_check)
                }
            }
            assert_eq!(bucket.regions().len(), 0);

            // Multiple allocations of different alignment.
            for size in sizes {
                for (i, align) in alignments.iter().enumerate() {
                    allocations[i] = allocate_aligned(&mut bucket, size, *align, corruption_check);
                }
                for allocation in allocations {
                    deallocate_aligned(&mut bucket, allocation, corruption_check)
                }
            }
            assert_eq!(bucket.regions().len(), 0);
        }
    }

    #[test]
    fn shrink() {
        unsafe {
            let mut corruption_check = 42;
            let mut bucket = Bucket::new();

            // Allocate entire page.
            let first_layout =
                Layout::array::<u8>(page_size() - REGION_HEADER_SIZE - BLOCK_HEADER_SIZE).unwrap();
            let mut first_addr = bucket.allocate(first_layout).unwrap();

            first_addr.as_mut().fill(corruption_check);

            // Now shrink by half
            let first_layout_shrunk =
                Layout::from_size_align(first_layout.size() / 2, first_layout.align()).unwrap();
            let first_addr_shrunk = bucket
                .reallocate(&Realloc::shrink(
                    first_addr.cast(),
                    first_layout,
                    first_layout_shrunk,
                ))
                .unwrap();

            check_mem_corruption(first_addr_shrunk.as_ref(), corruption_check);

            // Shrinking by half should have shrunk the block in place
            assert_eq!(first_addr.cast::<u8>(), first_addr_shrunk.cast::<u8>());
            let first_region = bucket.regions().first().unwrap();
            assert_eq!(
                first_region.as_ref().first_block().as_ref().size(),
                first_layout_shrunk
                    .align_to(alignment::POINTER_SIZE)
                    .unwrap()
                    .pad_to_align()
                    .size()
            );
            assert_eq!(first_region.as_ref().num_blocks(), 2);
            assert_eq!(bucket.free_blocks.len(), 1);
            assert_eq!(bucket.regions.len(), 1);

            // Let's allocate the remaining block
            let second_layout = Layout::array::<u8>(
                page_size()
                    - REGION_HEADER_SIZE
                    - 2 * BLOCK_HEADER_SIZE
                    - first_region.as_ref().first_block().as_ref().size(),
            )
            .unwrap();

            let mut second_addr = bucket.allocate(second_layout).unwrap();
            assert_eq!(second_addr.as_ref().len(), second_layout.size());
            assert_eq!(first_region.as_ref().num_blocks(), 2);
            assert_eq!(bucket.free_blocks.len(), 0);

            corruption_check -= 10;
            second_addr.as_mut().fill(corruption_check);

            // Let's use page size alignment to force reallocation, because
            // we don't know what address mmap gave us for the region, we only
            // know that it is aligned to page size. The next address aligned
            // to page size is at the end of this region, so this should
            // reallocate.
            let second_layout_page_aligned =
                Layout::from_size_align(second_layout.size(), page_size()).unwrap();
            let second_addr_page_aligned = bucket
                .reallocate(&Realloc::shrink(
                    second_addr.cast(),
                    second_layout,
                    second_layout_page_aligned,
                ))
                .unwrap();
            let second_region = bucket.regions.last().unwrap();
            assert_eq!(bucket.regions.len(), 2);
            assert_eq!(second_region.as_ref().num_blocks(), 2);

            // Last block that we were using was deallocated, and the new one
            // won't take up all the space in the region.
            assert_eq!(bucket.free_blocks.len(), 2);
            assert_ne!(
                second_addr.as_mut_ptr(),
                second_addr_page_aligned.as_mut_ptr()
            );
            // Account for padding.
            assert_eq!(
                second_addr_page_aligned.len(),
                page_size() - REGION_HEADER_SIZE - BLOCK_HEADER_SIZE - second_layout.size()
            );
            assert_eq!(
                second_addr_page_aligned.as_mut_ptr() as usize % page_size(),
                0
            );

            // We've only filled the memory the allocator gave us the first
            // time, but after forcing reallocation to page size alignment it
            // will give us a little bit more than we need because the minimum
            // block size has to account for the padding needed in the worst
            // case.
            check_mem_corruption(
                &second_addr_page_aligned.as_ref()[0..second_addr.len()],
                corruption_check,
            );

            // Now let's reduce alignment to force everyhing to be moved
            // backwards.
            let previous_block_size = second_region.as_ref().first_block().as_ref().size();
            let second_layout_half_page_aligned =
                Layout::from_size_align(second_layout.size(), page_size() / 2).unwrap();
            let second_addr_half_page_aligned = bucket
                .reallocate(&Realloc::shrink(
                    second_addr_page_aligned.cast(),
                    second_layout_page_aligned,
                    second_layout_half_page_aligned,
                ))
                .unwrap();

            assert!(
                second_addr_half_page_aligned
                    .as_mut_ptr()
                    .offset_from(second_addr_page_aligned.as_mut_ptr())
                    < 0
            );
            // Less padding, so block is smaller.
            assert!(previous_block_size > second_region.as_ref().first_block().as_ref().size());
            assert_eq!(
                second_addr_half_page_aligned.as_mut_ptr() as usize % (page_size() / 2),
                0
            );
            check_mem_corruption(
                &second_addr_half_page_aligned.as_ref()[0..second_addr.len()],
                corruption_check,
            );

            bucket.deallocate(first_addr_shrunk.cast(), first_layout_shrunk);
            bucket.deallocate(
                second_addr_half_page_aligned.cast(),
                second_layout_half_page_aligned,
            );
            // No more regions, we've only worked with 2 pointers so far.
            assert_eq!(bucket.regions.len(), 0);

            // Last but not least, let's try to force everything to be moved
            // forward instead of backwards. For that, we are going to allocate
            // an entire page and then increase alignment but decrease size.
            let third_layout = Layout::from(first_layout);
            let mut third_addr = bucket.allocate(third_layout).unwrap();
            corruption_check += 15;
            third_addr.as_mut().fill(corruption_check);

            let third_layout_aligned_to_half_page =
                Layout::from_size_align(8, page_size() / 2).unwrap();
            let third_addr_aligned_to_half_page = bucket
                .reallocate(&Realloc::shrink(
                    third_addr.cast(),
                    third_layout,
                    third_layout_aligned_to_half_page,
                ))
                .unwrap();

            check_mem_corruption(
                &third_addr_aligned_to_half_page.as_ref()
                    [..third_layout_aligned_to_half_page.size()],
                corruption_check,
            );
            assert_eq!(bucket.regions.len(), 1);
            assert_eq!(bucket.free_blocks.len(), 1);
            assert_eq!(
                third_addr_aligned_to_half_page.as_mut_ptr() as usize % (page_size() / 2),
                0
            );
            assert!(
                third_addr_aligned_to_half_page
                    .as_mut_ptr()
                    .offset_from(third_addr.as_mut_ptr())
                    > 0
            );
            bucket.deallocate(
                third_addr_aligned_to_half_page.cast(),
                third_layout_aligned_to_half_page,
            );
        }
    }

    #[test]
    fn grow_by_consuming_next_or_prev() {
        unsafe {
            let mut bucket = Bucket::new();

            let first_layout = Layout::from_size_align(MIN_BLOCK_SIZE, 4).unwrap();
            let mut first_addr = bucket.allocate(first_layout).unwrap();
            let mut corruption_check = 200;

            first_addr.as_mut().fill(corruption_check);
            let first_region = bucket.regions.first().unwrap();

            // Let's test growth by consuming next block
            let first_layout_grow_to_40 = Layout::from_size_align(40, 4).unwrap();
            let first_addr_grow_to_40 = bucket
                .reallocate(&Realloc::grow(
                    first_addr.cast(),
                    first_layout,
                    first_layout_grow_to_40,
                ))
                .unwrap();

            assert_eq!(bucket.regions.len(), 1);
            assert_eq!(first_region.as_ref().num_blocks(), 2);
            assert_eq!(bucket.free_blocks.len(), 1);
            assert_eq!(first_addr_grow_to_40.len(), 40);
            assert_eq!(first_addr.as_mut_ptr(), first_addr_grow_to_40.as_mut_ptr());
            check_mem_corruption(
                &first_addr_grow_to_40.as_ref()[..first_layout.size()],
                corruption_check,
            );

            // Now let's try to consume the previous block
            let second_layout = Layout::array::<u8>(
                page_size()
                    - REGION_HEADER_SIZE
                    - 3 * BLOCK_HEADER_SIZE
                    - first_region.as_ref().first_block().as_ref().size(),
            )
            .unwrap();
            let mut second_addr = bucket.allocate(second_layout).unwrap();

            corruption_check = 3;
            second_addr.as_mut().fill(corruption_check);

            // This should set the first block free.
            bucket.deallocate(first_addr_grow_to_40.cast(), first_layout_grow_to_40);
            assert_eq!(bucket.free_blocks.len(), 1);

            let second_layout_grow_to_page_size =
                Layout::array::<u8>(page_size() - REGION_HEADER_SIZE - BLOCK_HEADER_SIZE).unwrap();

            let second_addr_grow_to_page_size = bucket
                .reallocate(&Realloc::grow(
                    second_addr.cast(),
                    second_layout,
                    second_layout_grow_to_page_size,
                ))
                .unwrap();

            // Should be the same as the first one because everything is moved
            // to the first block again.
            assert_eq!(
                first_addr.as_mut_ptr(),
                second_addr_grow_to_page_size.as_mut_ptr()
            );
            // We're using all the page, so no free blocks.
            assert_eq!(bucket.free_blocks.len(), 0);
            assert_eq!(first_region.as_ref().num_blocks(), 1);
            check_mem_corruption(
                &second_addr_grow_to_page_size.as_ref()[..second_addr.as_ref().len()],
                corruption_check,
            );

            bucket.deallocate(
                second_addr_grow_to_page_size.cast(),
                second_layout_grow_to_page_size,
            );
        }
    }

    #[test]
    fn grow_by_consuming_next_and_prev() {
        unsafe {
            let mut bucket = Bucket::new();

            // Let's test the final case of growing blocks
            let surrounding_blocks_layout = Layout::from_size_align(MIN_BLOCK_SIZE, 4).unwrap();
            let block_in_the_middle_layout = Layout::array::<u8>(
                page_size() - REGION_HEADER_SIZE - 3 * BLOCK_HEADER_SIZE - 2 * MIN_BLOCK_SIZE,
            )
            .unwrap();

            let first_addr = bucket.allocate(surrounding_blocks_layout).unwrap();
            let mut second_addr = bucket.allocate(block_in_the_middle_layout).unwrap();
            let third_addr = bucket.allocate(surrounding_blocks_layout).unwrap();

            // We've alredy tested allocations, but there should be 3 blocks and
            // 0 free blocks.
            let region = bucket.regions.first().unwrap();
            assert_eq!(region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 0);

            // Now this should construct the pattern we want to test
            bucket.deallocate(first_addr.cast(), surrounding_blocks_layout);
            bucket.deallocate(third_addr.cast(), surrounding_blocks_layout);
            assert_eq!(bucket.free_blocks.len(), 2);

            let corruption_check = 125;
            second_addr.as_mut().fill(corruption_check);

            let grow_layout = Layout::array::<u8>(
                block_in_the_middle_layout.size() + 2 * MIN_BLOCK_SIZE + 2 * BLOCK_HEADER_SIZE,
            )
            .unwrap();
            let second_addr_grow = bucket
                .reallocate(&Realloc::grow(
                    second_addr.cast(),
                    block_in_the_middle_layout,
                    grow_layout,
                ))
                .unwrap();

            // Only one block, no free blocks
            assert_eq!(region.as_ref().num_blocks(), 1);
            assert_eq!(bucket.free_blocks.len(), 0);

            // We're back at the beginning again
            assert_eq!(second_addr_grow.as_mut_ptr(), first_addr.as_mut_ptr());

            check_mem_corruption(
                &second_addr_grow.as_ref()[..second_addr.as_ref().len()],
                corruption_check,
            );
        }
    }
}
