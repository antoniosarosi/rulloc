use std::{alloc::Layout, mem, ptr::NonNull};

use crate::{
    block::{Block, BLOCK_HEADER_SIZE, MIN_BLOCK_SIZE},
    header::Header,
    list::LinkedList,
    platform,
};

/// Region header size in bytes. See [`Header<T>`] and [`Region`].
pub(crate) const REGION_HEADER_SIZE: usize = mem::size_of::<Header<Region>>();

/// Memory region specific data. All headers are also linked lists nodes, see
/// [`Header<T>`] and [`Block`]. In this case, a complete region header would be
/// [`Header<Region>`].
///
/// We use [`libc::mmap`] to request memory regions from the kernel, and we
/// cannot assume that these regions are adjacent because `mmap` might be used
/// outside of this allocator (and that's okay) or we unmapped a previously
/// mapped region, which causes its next and previous regions to be
/// non-adjacent. Therefore, we store regions in a linked list. Each region also
/// contains a linked list of blocks. This is the high level overview:
///
/// ```text
/// +--------+------------------------+      +--------+-------------------------------------+
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// | Region | | Block | -> | Block | | ---> | Region | | Block | -> | Block | -> | Block | |
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// +--------+------------------------+      +--------+-------------------------------------+
/// ```
pub(crate) struct Region {
    /// Blocks contained within this memory region.
    pub blocks: LinkedList<Block>,
    /// Size of the region excluding [`Header<Region>`] size.
    pub size: usize,
}

impl Header<Region> {
    /// Returns a pointer to the first block in this region.
    ///
    /// # Safety
    ///
    /// There is **ALWAYS** at least one block in the region.
    #[inline]
    pub unsafe fn first_block(&self) -> NonNull<Header<Block>> {
        self.data.blocks.first().unwrap_unchecked()
    }

    /// Region size excluding [`REGION_HEADER_SIZE`].
    #[inline]
    pub fn size(&self) -> usize {
        self.data.size
    }

    /// Region size including [`REGION_HEADER_SIZE`].
    #[inline]
    pub fn total_size(&self) -> usize {
        REGION_HEADER_SIZE + self.data.size
    }

    /// Number of blocks in this region.
    #[inline]
    pub fn num_blocks(&self) -> usize {
        self.data.blocks.len()
    }
}

/// Calculates the length in bytes that we should call `mmap` with if we
/// want to safely store at least `size` bytes.
///
/// # Arguments
///
/// * `size` - Amount of bytes that need to be allocated without including
/// any header. This value must be **already aligned**.
///
pub(crate) fn determine_region_length(size: usize) -> usize {
    // We'll store at least one block in this region, so we need space for
    // region header, block header and user content.
    let total_size = REGION_HEADER_SIZE + BLOCK_HEADER_SIZE + size;

    // Align up to page size. If we want to store 4104 bytes and page size is
    // 4096 bytes, then we'll request a region that's 2 pages in length
    // (8192 bytes).
    let mut length = Layout::from_size_align(total_size, platform::page_size())
        .unwrap()
        .pad_to_align()
        .size();

    // There's a little detail left. Whenever we request a new region using
    // mmap, we initialize the region with one single block that takes up all
    // the space except for headers. Later in the allocation process, if the
    // block is too big, it will be split in two different blocks. You can check
    // the code at [`crate::bucket`] for that.
    //
    // Now, if the total size needed to store region header, block header and
    // requested size is less than the region length there's a possibilty that
    // the resulting block cannot be split in 2 different blocks. Imagine that
    // the requested size is 3960 bytes and we are running on a 64 bit machine.
    // On such machine, REGION_HEADER_SIZE = 48 and BLOCK_HEADER_SIZE = 40. That
    // makes total_size = 4048 bytes. If the page size is 4096 bytes, then we
    // would request a memory region that's 4096 bytes in length because of the
    // alignment, but the big block in the region cannot be split in two because
    // those 48 bytes that we have left (4096 - 4048) cannot fit a minimum
    // block. On 64 bit machines, this is what happens:
    //
    // +----------+---------------------------------------+
    // |          | +--------+--------------------------+ |
    // |  Region  | | Block  | Block           48 bytes | |
    // |  Header  | | Header | Content          Wasted  | |
    // |          | +--------+--------------------------+ |
    // +----------+---------------------------------------+
    // ^          ^ ^        ^                          ^
    // |          | |        |                          |
    // +----------+ +--------+--------------------------+
    //   48 bytes    40 bytes          4008 bytes
    // ^                                                  ^
    // |                                                  |
    // +--------------------------------------------------+
    //                       4096 bytes
    //
    // The block created in the region can store 4008 bytes of content, but 48
    // of those are wasted because the user only needs 3960 bytes and we cannot
    // create a new block from them either, because it would only fit the header
    // without any content. So we have two options: either waste those bytes and
    // wait until the block is deallocated or reallocated for them to be used
    // again, or request an additional page to make sure the block can be
    // split in two.
    //
    // For now, we'll just request another page so that we have a free block,
    // but the other option isn't actually bad. The same scenario can happen
    // when searching for free blocks, we might find one that can fit the
    // requested size but wastes a little space because we can't split it, so
    // this will only help reduce fragmentation when mapping new regions, but
    // anything can happen from there on.
    if total_size < length && total_size + BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE > length {
        length += platform::page_size();
    }

    length
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{alignment::POINTER_SIZE, platform::PAGE_SIZE};

    #[test]
    fn region_length() {
        unsafe {
            // Basic checks.
            assert_eq!(determine_region_length(POINTER_SIZE), platform::page_size());
            assert_eq!(determine_region_length(PAGE_SIZE / 2), PAGE_SIZE);
            for i in 1..=100 {
                assert_eq!(determine_region_length(PAGE_SIZE * i), PAGE_SIZE * (i + 1));
            }

            // Some corner cases.
            let exact_remaining_space = PAGE_SIZE - REGION_HEADER_SIZE - BLOCK_HEADER_SIZE;
            assert_eq!(determine_region_length(exact_remaining_space), PAGE_SIZE);

            let enough_space_for_minimum_block_at_the_end =
                PAGE_SIZE - REGION_HEADER_SIZE - 2 * BLOCK_HEADER_SIZE - MIN_BLOCK_SIZE;
            assert_eq!(
                determine_region_length(enough_space_for_minimum_block_at_the_end),
                PAGE_SIZE
            );

            let not_enough_space_for_minimum_block_at_the_end =
                PAGE_SIZE - REGION_HEADER_SIZE - 2 * BLOCK_HEADER_SIZE - MIN_BLOCK_SIZE
                    + POINTER_SIZE;
            assert_eq!(
                determine_region_length(not_enough_space_for_minimum_block_at_the_end),
                2 * PAGE_SIZE
            );
        }
    }
}
