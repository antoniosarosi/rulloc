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
///                  +------------------------------------+   +-----------------------+
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
    pub unsafe fn allocate(&mut self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
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

            munmap(region.cast(), region.as_ref().total_size());
        }
    }

    /// Makes the allocated block smaller. Shrinking is always done in place
    /// unless we can't satisfy the new alignment constraints. See
    /// [`crate::alignment`].
    pub unsafe fn shrink(
        &mut self,
        address: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let block = Header::<Block>::from_allocated_pointer(address, old_layout);
        let content_addr = Header::<Block>::content_address_of(block);

        // We'll prioritize block reusability where possible, so don't add
        // extra padding for alignment. We'll deal with padding later.
        let new_size = alignment::minimum_block_size_excluding_padding(new_layout);

        // Best case scenario, we can shrink this block in place without doing
        // much work, so early return.
        if new_layout.align() <= alignment::POINTER_SIZE {
            // If the old alignment was greater than pointer size then get rid
            // of padding to reduce fragmentation.
            if old_layout.align() > alignment::POINTER_SIZE {
                ptr::copy(address.as_ptr(), content_addr.as_ptr(), new_layout.size());
            }
            self.split_block_if_possible(block, new_size);

            return Ok(NonNull::slice_from_raw_parts(
                content_addr,
                block.as_ref().size(),
            ));
        }

        // The code below deals with all the rest of cases. We know for sure
        // that the new layout needs padding, old layout might have had padding
        // or not but we don't care because we're already given the address
        // where to user content starts in case we need to copy it somewhere
        // else.
        let (next_aligned, padding) = alignment::next_aligned(content_addr, new_layout.align());

        // Can't reuse this block, so find a new one and return.
        if padding + new_size > block.as_ref().size() {
            return Ok(self.reallocate(block, address, old_layout, new_layout)?);
        }

        // We only need to copy the contents if the current address is
        // not already aligned. If `next_aligned` is located before `address`
        // then the alignment has decreased so by moving the content backwards
        // we'll reduce fragmentation. If `next_aligned` is located after
        // `address` then the alignment has increased but we can still reuse
        // this block because it fits the new padding. Otherwise, the address
        // stays the same whether or not the alignment has changed, because it
        // is already aligned.
        //
        // Note that if old alignment was POINTER_SIZE this still works because
        // the `next_aligned()` function will never return the content address
        // of the block, so we're safe, the alignment back pointer won't
        // override fields of the block header. Of course, this conclusion was
        // reached by first writing a dozen of if-else statements and noticing
        // that the same code is repeated everywhere.
        if next_aligned != address {
            ptr::copy(address.as_ptr(), next_aligned.as_ptr(), new_layout.size());
            ptr::write(alignment::back_pointer_of(next_aligned).as_ptr(), block);
        }

        self.split_block_if_possible(block, new_size + padding);

        Ok(NonNull::slice_from_raw_parts(
            next_aligned,
            block.as_ref().size() - padding,
        ))
    }

    /// Reallocates the contents of a block somewhere else. This function should
    /// only be called if a block cannot be reused for shrinking or growing.
    /// Initial content address from where data should be copied to the new
    /// block has to be provided. The given block will be automatically added
    /// to the free list.
    unsafe fn reallocate(
        &mut self,
        block: NonNull<Header<Block>>,
        address: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new_address = self.allocate(new_layout)?;
        ptr::copy_nonoverlapping(
            address.as_ptr(),
            new_address.as_mut_ptr(),
            old_layout.size(),
        );
        self.free_blocks.append_block(block);

        Ok(new_address)
    }

    /// This function performs the algorithm described at
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
        self.free_blocks
            .iter_blocks()
            .find(|block| block.as_ref().size() >= size)
    }

    /// Block splitting algorithm implementation. Let's say we have a free block
    /// that can hold 128 bytes and a request to allocate 8 bytes has been made.
    /// We'll split the free block in two different blocks, like so:
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
    ///         |     |  Content  | <- 8 bytes.
    ///         +-->  +-----------+
    ///         |     |   Header  | <- H bytes.
    /// Block 2 |     +-----------+
    ///         |     |  Content  | <- 128 bytes - 8 bytes - H bytes.
    ///         +-->  +-----------+
    /// ```
    ///
    /// The block doesn't necessarily have to be free, it might be in use but
    /// we want to shrink it. See [`Self::shrink`]. This function does not touch
    /// the contents of the block, it only changes it's header to reflect the
    /// new size. On the other hand, the new block created in the splitting
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
    /// Note that only one of the surrounding is merged if the other one is not
    /// free. Also, if the previous block is merged, then the address of the
    /// current block changes. That's why we have to return a pointer to a
    /// block.
    ///
    /// # Safety
    ///
    /// Unlike [`Self::split_free_block`], the caller must guarantee that
    /// `block` is free in this case.
    #[rustfmt::skip]
    unsafe fn merge_surrounding_free_blocks_if_possible(
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

impl Drop for Bucket {
    fn drop(&mut self) {
        for region in &*self.regions {
            unsafe { munmap(region.cast(), region.as_ref().total_size()) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        alignment::AlignmentBackPointer,
        region::{page_size, PAGE_SIZE},
    };

    #[test]
    fn allocs_and_deallocs() {
        unsafe {
            let mut bucket = Bucket::new();

            // Request 1 byte, should call `mmap` with length of PAGE_SIZE.
            let first_layout = Layout::new::<u8>();
            let first_addr = bucket.allocate(first_layout).unwrap().cast::<u8>();

            // We'll use this later to check memory corruption. The allocator
            // should not touch the content of any block.
            let corruption_check = 69;
            *first_addr.as_ptr() = corruption_check;

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
            let second_addr = bucket.allocate(second_layout).unwrap().cast::<u8>();

            // We'll check corruption later.
            for i in 0..second_layout.size() {
                *second_addr.as_ptr().add(i) = corruption_check;
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

            for i in 0..third_layout.size() {
                *third_addr.as_ptr().add(i) = corruption_check;
            }

            // Number of blocks hasn't changed, but we don't have free blocks
            // anymore.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(bucket.free_blocks.len(), 0);

            // Time for checking memory corruption
            assert_eq!(*first_addr.as_ptr(), corruption_check);
            for i in 0..second_layout.size() {
                assert_eq!(*second_addr.as_ptr().add(i), corruption_check);
            }
            for i in 0..third_layout.size() {
                assert_eq!(*third_addr.as_ptr().add(i), corruption_check);
            }

            // Let's request a bigger chunk so that a new region is used.
            let fourth_layout = Layout::array::<u8>(PAGE_SIZE * 2 - PAGE_SIZE / 2).unwrap();
            let fourth_addr = bucket.allocate(fourth_layout).unwrap().cast::<u8>();

            for i in 0..fourth_layout.size() {
                *fourth_addr.as_ptr().add(i) = corruption_check;
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
            for i in 0..fourth_layout.size() {
                assert_eq!(*fourth_addr.as_ptr().add(i), corruption_check);
            }

            // Deallocating fourh address should unmap the last region.
            bucket.deallocate(fourth_addr, fourth_layout);
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
        let addr = bucket.allocate(layout).unwrap().cast::<u8>();

        // We are not actually performing aligned memory accesses,
        // but it doesn't matter, we just wanna check that we can
        // write to the requested memory and we don't seg fault.
        // We're not writing the entire layout when using Miri
        // because it's too slow, we'll just write to the addresses
        // that might cause problems, and Miri wll catch bugs or
        // undefined behaviour.
        if cfg!(miri) {
            *addr.as_ptr() = corruption_check;
            *addr.as_ptr().add(size / 2) = corruption_check;
            *addr.as_ptr().add(size - 1) = corruption_check;
        } else {
            for offset in 0..size {
                *addr.as_ptr().add(offset) = corruption_check;
            }
        }

        assert_eq!(addr.cast::<u8>().as_ptr() as usize % align, 0);

        (addr, layout)
    }

    unsafe fn deallocate_aligned(
        bucket: &mut Bucket,
        aligned_alloc: (NonNull<u8>, Layout),
        corruption_check: u8,
    ) {
        let (addr, layout) = aligned_alloc;
        if cfg!(miri) {
            assert_eq!(*addr.as_ptr(), corruption_check);
            assert_eq!(*addr.as_ptr().add(layout.size() / 2), corruption_check);
            assert_eq!(*addr.as_ptr().add(layout.size() - 1), corruption_check);
        } else {
            for offset in 0..layout.size() {
                assert_eq!(*addr.as_ptr().add(offset), corruption_check);
            }
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

    fn check_mem_corruption(chunk: &[u8], corruption_check: u8) {
        for value in chunk {
            assert_eq!(value, &corruption_check);
        }
    }

    #[test]
    fn shrink() {
        unsafe {
            let corruption_check = 42;
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
                .shrink(first_addr.cast(), first_layout_shrunk, first_layout_shrunk)
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

            second_addr.as_mut().fill(corruption_check);

            // Let's use page size alignment to force reallocation, because
            // we don't know what address mmap gave us for the region, we only
            // know that it is aligned to page size. The next address aligned
            // to page size is at the end of this region, so this should
            // reallocate.
            let second_layout_page_aligned =
                Layout::from_size_align(second_layout.size(), page_size()).unwrap();
            let second_addr_page_aligned = bucket
                .shrink(
                    second_addr.cast(),
                    second_layout,
                    second_layout_page_aligned,
                )
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
                .shrink(
                    second_addr_page_aligned.cast(),
                    second_layout_page_aligned,
                    second_layout_half_page_aligned,
                )
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
        }
    }
}
