use std::{mem, ptr::NonNull};

use crate::{
    block::{Block, BLOCK_HEADER_SIZE, MIN_BLOCK_SIZE},
    header::Header,
    list::LinkedList,
};

/// Region header size in bytes. See [`Header<T>`] and [`Region`].
pub(crate) const REGION_HEADER_SIZE: usize = mem::size_of::<Header<Region>>();

/// Virtual memory page size. 4096 bytes on most computers. This should be a
/// constant but we don't know the value at compile time.
pub(crate) static mut PAGE_SIZE: usize = 0;

/// We only know the value of the page size at runtime by calliing
/// [`libc::sysconf`], so we'll call that function once and then mutate a global
/// variable to reuse it.
#[inline]
pub(crate) unsafe fn page_size() -> usize {
    if PAGE_SIZE == 0 {
        PAGE_SIZE = libc::sysconf(libc::_SC_PAGE_SIZE) as usize
    }

    PAGE_SIZE
}

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
/// +--------+------------------------+      ---------+-------------------------------------+
/// ```
#[derive(Clone, Copy, Debug)]
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
    pub unsafe fn first_block(&self) -> NonNull<Header<Block>> {
        self.data.blocks.first().unwrap_unchecked()
    }

    /// Region size excluding [`REGION_HEADER_SIZE`].
    pub fn size(&self) -> usize {
        self.data.size
    }

    /// Region size including [`REGION_HEADER_SIZE`].
    pub fn total_size(&self) -> usize {
        REGION_HEADER_SIZE + self.data.size
    }

    /// Number of blocks in this region.
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
pub(crate) unsafe fn determine_region_length(size: usize) -> usize {
    // We'll store at least one block in this region, so we need space for
    // region header, block header and user content.
    let total_size = REGION_HEADER_SIZE + BLOCK_HEADER_SIZE + size;

    // Force round up. If we want to store 4104 bytes and page size is 4096
    // bytes, then we'll request a region that's 2 pages in length
    // (8192 bytes).
    let mut length = page_size() * ((total_size + page_size() - 1) / page_size());

    // If the total size is smaller than the region length we want to make
    // sure that at least one more block can be allocated in that region.
    // Imagine that total size ends up being 4048 bytes and page size is
    // 4096 bytes. We would request a memory region that's 4096 bytes in
    // length because of the round up, but we would end up with a little
    // chunk of 48 bytes at the end of the region where we cannot allocate
    // anything because block header + minimum block size doesn't fit in 48
    // bytes. At least on 64 bit machines.
    //
    // +-----------------+
    // | Region header   | <---+
    // +-----------------+     |
    // | Block header    |     | This is 4048 bytes, total length is 4096.
    // +-----------------+     |
    // | Content         | <---+
    // +-----------------+
    // | Remaining chunk | <- 48 bytes. Nothing can be allocated here.
    // +-----------------+
    //
    // If that happens, we'll request an extra 4096 bytes or whatever the
    // page size is. The other possible case is that total_size is exactly
    // equal to length. That's okay because the region would have only one
    // block that doesn't waste any space and the entire region would be
    // returned back to the kernel when the block is deallocated.
    if total_size < length && total_size + BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE > length {
        length += page_size();
    }

    length
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_length() {
        use crate::bucket::POINTER_SIZE;

        unsafe {
            // Basic checks.
            assert_eq!(determine_region_length(POINTER_SIZE), page_size());
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
