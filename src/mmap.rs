use std::{alloc::Layout, mem, ptr};

use libc;

use crate::align;

/// Minimum block size in bytes. Read the documentation of the data structures
/// used for this allocator to understand why it has this value.
const MIN_BLOCK_SIZE: usize = mem::size_of::<FreeBlockLinks>();

/// Virtual memory page size. 4096 bytes on most computers. This should be a
/// constant but we don't know the value at compile time.
static mut PAGE_SIZE: usize = 0;

/// We only know the value of the page size at runtime by calliing `sysconf`,
/// so we'll call that function once and then mutate a global variable to reuse
/// it.
#[inline]
unsafe fn page_size() -> usize {
    if PAGE_SIZE == 0 {
        PAGE_SIZE = libc::sysconf(libc::_SC_PAGE_SIZE) as usize
    }

    PAGE_SIZE
}

/// Memory block header. Here's a graphical representation of how it looks like
/// in memory:
///
/// ```text
/// +--------------------------+
/// | pointer to block region  |   <------+
/// +--------------------------+          |
/// | pointer to next block    |          |
/// +--------------------------+          |
/// | pointer to prev block    |          |
/// +--------------------------+          | Block struct (This is the header).
/// | block size               |          |
/// +--------------------------+          |
/// | is free flag             |          |
/// +--------------------------+          |
/// | padding (word alignment) |   <------+
/// +--------------------------+
/// |      User content        |   <------+
/// |           ...            |          |
/// |           ...            |          | This is where the user writes stuff.
/// |           ...            |          |
/// |           ...            |   <------+
/// +--------------------------+
/// ```
#[derive(Debug)]
struct Block {
    /// Memory region where this block is located.
    region: *mut Region,
    /// Next block in this region. Null if this is the last one.
    next: *mut Block,
    /// Previous block in this region. Null if this is the first one.
    prev: *mut Block,
    /// Size of the block excluding size of header.
    size: usize,
    /// Whether this block can be used or not.
    is_free: bool,
}

/// Memory region header. `mmap` gives us memory regions that are not
/// necessarily one after the other, so they don't follow a particular order. We
/// store them as a linked list, and each region contains a list of blocks
/// within itself.
///
/// ```text
/// +--------+------------------------+      +--------+-------------------------------------+
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// | Region | | Block | -> | Block | | ---> | Region | | Block | -> | Block | -> | Block | |
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// +--------+------------------------+      ---------+-------------------------------------+
/// ```
#[derive(Debug)]
struct Region {
    /// Total size of the region, this is the length we called `mmap` with. So
    /// it includes everything, from headers to content.
    length: usize,
    /// Amount of bytes in this region used by the allocator. Free list doesn't
    /// count because it reuses content space.
    used_by_allocator: usize,
    /// Amount of bytes in this region used by the user.
    used_by_user: usize,
    /// Number of blocks in this region.
    num_blocks: usize,
    /// Next region. Null if this is the last one.
    next: *mut Region,
    /// Previous region. Null if this is the first one.
    prev: *mut Region,
}

/// When a block is free we'll use the content to store a free list, that is, a
/// linked list of _only_ free blocks. Since we want a doubly linked list, we
/// need to store 2 pointers, one for the previous block and another one for
/// the next free block. This is how a free block would look like in memory:
///
/// ```text
/// +----------------------------+ <- Block struct starts here.
/// |   Header (Block struct)    |
/// +----------------------------+ <- FreeBlockLinks struct starts here.
/// | pointer to next free block |
/// +----------------------------+
/// | pointer to prev free block |
/// +----------------------------+ <- FreeBlockLinks struct ends here.
/// |   Rest of user content     | <- This could be 0 bytes.
/// |          ......            |
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
#[derive(Debug)]
struct FreeBlockLinks {
    /// Next free block. This could point to a block located in the same region
    /// as this one or a block located on a different region.
    next: *mut Block,
    /// Previous free block. This could point to a block located in the same
    /// region as this one or a block located on a different region.
    prev: *mut Block,
}

/// General purpose allocator. All memory is requested from the kernel using
/// `mmap` and some tricks and optimizations are implemented such as free list,
/// block coalescing and block splitting.
#[derive(Debug)]
pub struct MmapAllocator {
    /// First available block. Could be located in any region.
    first_free_block: *mut Block,
    /// Last available block. Could be located in any region.
    last_free_block: *mut Block,
    /// Number of free blocks in the free list.
    num_free_blocks: usize,
    /// First region that we've allocated with `mmap`.
    first_region: *mut Region,
    /// Last region allocated with `mmap`.
    last_region: *mut Region,
    /// Number of memory regions in the linked list.
    num_regions: usize,
}

impl Block {
    /// Returns the block header located before `address`.
    ///
    /// # Arguments
    ///
    /// * `address` - Address of the block content. If this is not exactly
    /// the address located after the block header then expect undefined
    /// behaviour.
    pub unsafe fn from_content_address(address: *mut u8) -> *mut Self {
        address.sub(mem::size_of::<Self>()) as *mut Self
    }

    /// Total size of the block including header size.
    pub unsafe fn total_size(&self) -> usize {
        mem::size_of::<Self>() + self.size
    }

    /// Returns the address after the header.
    ///
    /// ```text
    /// +---------+ <- Block struct (header) starts here.
    /// | Header  |
    /// +---------+ <- User content starts here.
    /// | Content | <- Returned address points to the first cell after header.
    /// +---------+
    /// |   ...   |
    /// +---------+
    /// |   ...   |
    /// +---------+
    /// ```
    pub unsafe fn content_address(&self) -> *mut u8 {
        (self as *const Self).offset(1) as *mut u8
    }

    /// See [`FreeBlockLinks`] struct for details.
    pub unsafe fn free_block_links(&self) -> *mut FreeBlockLinks {
        self.content_address() as *mut FreeBlockLinks
    }

    /// Returns the pointer to the next free block. Free list is stored as
    /// pointers in the content part of free blocks.
    ///
    /// ```text
    /// +----------------------------+ <- Block struct starts here.
    /// |           Header           |
    /// +----------------------------+ <- User content starts here.
    /// | pointer to next free block | <- Pointer in this cell is returned.
    /// +----------------------------+
    /// | pointer to prev free block |
    /// +----------------------------+
    /// ```
    pub unsafe fn next_free(&self) -> *mut Self {
        (*self.free_block_links()).next
    }

    /// See [`FreeBlockLinks`] struct and [`Self::next_free`] method.
    pub unsafe fn set_next_free(&self, next_free: *mut Block) {
        (*self.free_block_links()).next = next_free;
    }

    /// Returns the pointer to the previous free block. Free list is stored as
    /// pointers in the content part of free blocks.
    ///
    /// ```text
    /// +----------------------------+ <- Block struct starts here.
    /// |           Header           |
    /// +----------------------------+ <- User content starts here.
    /// | pointer to next free block |
    /// +----------------------------+
    /// | pointer to prev free block | <- Pointer in this cell is returned.
    /// +----------------------------+
    /// ```
    pub unsafe fn prev_free(&self) -> *mut Self {
        (*self.free_block_links()).prev
    }

    /// See [`FreeBlockLinks`] struct and [`Self::prev_free`] method.
    pub unsafe fn set_prev_free(&self, prev_free: *mut Block) {
        (*self.free_block_links()).prev = prev_free;
    }
}

impl Region {
    /// Returns the address right after the region header.
    pub unsafe fn first_block_address(&self) -> *mut Block {
        (self as *const Self).offset(1) as *mut Block
    }
}

impl MmapAllocator {
    // Constructs a new allocator. No actual allocations happen until memory
    // is requested using [`MmapAllocator::alloc`].
    pub fn new() -> Self {
        Self {
            first_free_block: ptr::null_mut(),
            last_free_block: ptr::null_mut(),
            first_region: ptr::null_mut(),
            last_region: ptr::null_mut(),
            num_regions: 0,
            num_free_blocks: 0,
        }
    }

    /// Allocates a new block that can fit at least `layout.size()` bytes.
    /// Because of alignment and headers, it might allocate a bigger block than
    /// needed. As long as no more than `layout.size()` bytes are written on
    /// the content part of the block it should be fine.
    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let content_size = align(if layout.size() >= MIN_BLOCK_SIZE {
            layout.size()
        } else {
            MIN_BLOCK_SIZE
        });

        let mut free_block = self.find_free_block(content_size);

        // We found a free block, if it's too big we'll split it in two and
        // return to the user a new block that can only fit the requested size.
        if !free_block.is_null() {
            self.split_free_block_if_possible(free_block, content_size);
            self.remove_block_from_free_list(free_block);
            (*(*free_block).region).used_by_user += (*free_block).size;
            return (*free_block).content_address();
        }

        // Didn't find any free block, so fun stuff begins. First, request a new
        // memory region from the kernel.
        let region = self.request_region(content_size);

        if region.is_null() {
            return ptr::null_mut();
        }

        // Allocate first block in this region. We'll give this block to the
        // user.
        let block = (*region).first_block_address();
        *block = Block {
            region,
            size: content_size,
            is_free: false,
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        };

        // Update stats.
        (*region).used_by_allocator += mem::size_of::<Block>();
        (*region).used_by_user += (*block).size;
        (*region).num_blocks += 1;

        // We've allocated more than needed, so now we have a free block.
        if (*region).length > mem::size_of::<Region>() + mem::size_of::<Block>() + content_size {
            self.add_free_block_next_to(block);
        }

        (*block).content_address()
    }

    /// Deallocates the given pointer. Memory might not be returned to the OS
    /// if the region where `address` is located still contains used blocks.
    /// However, the freed block will be reused later if possible.
    pub unsafe fn dealloc(&mut self, address: *mut u8) {
        let mut block = Block::from_content_address(address);

        (*(*block).region).used_by_user -= (*block).size;

        self.append_block_to_free_list(block);

        // If left block is merged then the address will change.
        block = self.merge_free_blocks_if_possible(block);

        // All blocks have been merged into one, so we can return this region
        // back to the kernel.
        if (*(*block).region).num_blocks == 1 {
            // The free block in this region is no longer valid because this
            // region is about to be unmapped.
            self.remove_block_from_free_list(block);
            // Region has to be removed before unmapping, otherwise seg fault.
            self.remove_region((*block).region);
            let length = (*(*block).region).length as libc::size_t;
            if libc::munmap(address as *mut libc::c_void, length) != 0 {
                // TODO: What should we do here? Panic? Memory region is still
                // valid here, it wasn't unmapped.
            }
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
    unsafe fn determine_region_length(&self, size: usize) -> usize {
        // We'll store at least one block in this region, so we need space for
        // region header, block header and user content.
        let total_size = mem::size_of::<Region>() + mem::size_of::<Block>() + size;

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
        // bytes.
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
        if total_size < length && total_size + mem::size_of::<Block>() + MIN_BLOCK_SIZE > length {
            length += page_size();
        }

        length
    }

    /// Calls `mmap` and returns the resulting address or null if `mmap fails`.
    ///
    /// # Arguments
    ///
    /// * `length` - Length that we should call `mmap` with. This should be a
    /// multiple of [`PAGE_SIZE`].
    unsafe fn mmap(&self, length: usize) -> *mut u8 {
        // C void null pointer. This is what we need to request memory with mmap.
        let null = ptr::null_mut::<libc::c_void>();
        // Memory protection. Read-Write only.
        let protection = libc::PROT_READ | libc::PROT_WRITE;
        // Memory flags. Should be private to our process and not mapped to any
        // file or device (MAP_ANONYMOUS).
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

        match libc::mmap(null, length, protection, flags, -1, 0) {
            libc::MAP_FAILED => ptr::null_mut(),
            address => address as *mut u8,
        }
    }

    /// Requests a new memory region from the kernel where we can fit `size`
    /// bytes plus headers. See [`MmapAllocator::determine_region_length`].
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes (must be aligned to word size) that
    /// need to be allocated without including any headers.
    unsafe fn request_region(&mut self, size: usize) -> *mut Region {
        let length = self.determine_region_length(size);
        let region = self.mmap(length) as *mut Region;

        if region.is_null() {
            return ptr::null_mut();
        }

        *region = Region {
            length,
            next: ptr::null_mut(),
            prev: self.last_region,
            num_blocks: 0,
            used_by_allocator: mem::size_of::<Region>(),
            used_by_user: 0,
        };

        if self.first_region.is_null() {
            self.first_region = region;
        } else {
            (*self.last_region).next = region;
        }
        self.last_region = region;

        self.num_regions += 1;

        region
    }

    /// Removes region from linked list.
    unsafe fn remove_region(&mut self, region: *mut Region) {
        if self.num_regions == 1 {
            self.first_region = ptr::null_mut();
            self.last_region = ptr::null_mut();
        } else if region == self.first_region {
            (*(*region).next).prev = ptr::null_mut();
            self.first_region = (*region).next;
        } else if region == self.last_region {
            (*(*region).prev).next = ptr::null_mut();
            self.last_region = (*region).prev;
        } else {
            (*(*region).prev).next = (*(*region).next).next;
            (*(*region).next).prev = (*(*region).prev).prev;
        }

        self.num_regions -= 1;
    }

    /// Returns the first free block in the free list or null if none was found.
    unsafe fn find_free_block(&self, size: usize) -> *mut Block {
        let mut current = self.first_free_block;

        while !current.is_null() {
            if (*current).size >= size {
                return current;
            }
            current = (*current).next_free();
        }

        ptr::null_mut()
    }

    /// Fill the unused part of a region with a newly created free block. Should
    /// only be called if the region that holds the given `block` actually has
    /// an unused chunk at the end that can fit a new block of at least
    /// [`MIN_BLOCK_SIZE`]. See [`Self::determine_region_length`] for more
    /// details.
    unsafe fn add_free_block_next_to(&mut self, block: *mut Block) {
        // Remaining chunk size.
        let chunk_size =
            (*(*block).region).length - mem::size_of::<Region>() - (*block).total_size();

        // Write block.
        let free_block = (*block).content_address().add((*block).size) as *mut Block;
        *free_block = Block {
            region: (*block).region,
            is_free: true,
            next: ptr::null_mut(),
            prev: block,
            size: chunk_size - mem::size_of::<Block>(),
        };
        (*block).next = free_block;

        // Update stats.
        (*(*block).region).num_blocks += 1;
        (*(*block).region).used_by_allocator += mem::size_of::<Block>();

        // Update free list.
        self.append_block_to_free_list(free_block);
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
    unsafe fn split_free_block_if_possible(&mut self, block: *mut Block, size: usize) {
        // If there's not enough space available we can't split the block.
        if (*block).size < size + mem::size_of::<Block>() + MIN_BLOCK_SIZE {
            return;
        }

        // If we can, the block is located `size` bytes after the initial
        // content address of the current block.
        let new_block = (*block).content_address().add(size) as *mut Block;
        *new_block = Block {
            prev: block,
            next: (*block).next,
            size: (*block).size - size - mem::size_of::<Block>(),
            is_free: true,
            region: (*block).region,
        };
        (*block).next = new_block;

        // The current block can only hold `size` bytes from now on.
        (*block).size = size;

        // Update free list.
        self.append_block_to_free_list(new_block);

        // Update stats.
        (*(*block).region).num_blocks += 1;
        (*(*block).region).used_by_allocator += mem::size_of::<Block>();
    }

    /// This function performs the inverse of [`Self::split_free_block_if_possible`].
    /// If surrounding blocks are free, then we'll merge them all into one
    /// bigger block.
    ///
    /// **Before**:
    ///
    /// ```text
    ///                         +-->  +-----------+
    ///                         |     |   Header  |
    /// Block A, Free           |     +-----------+
    ///                         |     |  Content  | <- A bytes.
    ///                         +-->  +-----------+
    ///                         |     |   Header  |
    /// Block B, Recently freed |     +-----------+
    ///                         |     |  Content  | <- B bytes.
    ///                         +-->  +-----------+
    ///                         |     |   Header  |
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
    ///                         |     |   Header  |
    /// Block D, Bigger block   |     +-----------+
    ///                         |     |  Content  | <- A + B + C bytes.
    ///                         +-->  +-----------+
    /// ```
    unsafe fn merge_free_blocks_if_possible(&mut self, mut block: *mut Block) -> *mut Block {
        if !(*block).next.is_null() && (*(*block).next).is_free {
            self.merge_next_block(block);
        }

        if !(*block).prev.is_null() && (*(*block).prev).is_free {
            block = (*block).prev;
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
    unsafe fn merge_next_block(&mut self, block: *mut Block) {
        // First update free list. The new bigger block will become the
        // last block, and the 2 old smaller blocks will "dissapear" from the
        // list.
        self.remove_block_from_free_list((*block).next);
        self.remove_block_from_free_list(block);
        self.append_block_to_free_list(block);

        // Now this block is bigger.
        (*block).size += (*(*block).next).total_size();

        // Update local region block list.
        (*block).next = (*(*block).next).next;
        if !(*block).next.is_null() {
            (*(*block).next).prev = block;
        }
        // Previous block in the same region was already pointinig to
        // current block, so we don't have to update it.

        (*(*block).region).num_blocks -= 1;
        // We lost one header after merging the next block.
        (*(*block).region).used_by_allocator -= mem::size_of::<Block>();
    }

    /// This is basically the doubly linked list removal algorithm. Assumes
    /// that the given block is part of the free list. See [`FreeBlockLinks`]
    /// for more details. Expect undefined behaviour otherwise.
    unsafe fn remove_block_from_free_list(&mut self, block: *mut Block) {
        if self.num_free_blocks == 1 {
            self.last_free_block = ptr::null_mut();
            self.first_free_block = ptr::null_mut();
        } else if block == self.first_free_block {
            (*(*block).next_free()).set_prev_free(ptr::null_mut());
            self.first_free_block = (*block).next_free();
        } else if block == self.last_free_block {
            (*(*block).prev_free()).set_next_free(ptr::null_mut());
            self.last_free_block = (*block).prev_free();
        } else {
            (*(*block).prev_free()).set_next_free((*block).next_free());
            (*(*block).next_free()).set_prev_free((*block).prev_free());
        }

        (*block).is_free = false;
        self.num_free_blocks -= 1;
    }

    /// Append the given block to the free list. Undefined behaviour if the
    /// block content is used for anything other than pointers to next/prev
    /// free blocks after this action. See [`FreeBlockLinks`].
    unsafe fn append_block_to_free_list(&mut self, block: *mut Block) {
        (*block).set_next_free(ptr::null_mut());
        (*block).set_prev_free(self.last_free_block);

        if self.first_free_block.is_null() {
            self.first_free_block = block;
        } else {
            (*self.last_free_block).set_next_free(block);
        }
        self.last_free_block = block;

        (*block).is_free = true;
        self.num_free_blocks += 1;
    }

    /// Some quick and dirty statistics.
    pub unsafe fn print_stats(&self) {
        println!("{self:?}\n");

        let mut region = self.first_region;

        let mut i = 0;
        while !region.is_null() {
            println!(
                "Region {i}: {:?} - {:?} {:?}",
                region,
                (region as *mut u8).add((*region).length),
                *region
            );
            let mut j = 0;
            let mut block = (*region).first_block_address();
            while !block.is_null() {
                println!(
                    " Block {j}: {:?} - {:?}, {:?}",
                    block,
                    (block as *mut u8).add((*block).total_size()),
                    *block
                );
                block = (*block).next;
                j += 1;
            }
            println!(
                "Summary: Allocator: {:.2}%, User: {:.2}%, Free: {:.2}%\n",
                (*region).used_by_allocator as f64 / (*region).length as f64 * 100.0,
                (*region).used_by_user as f64 / (*region).length as f64 * 100.0,
                (1.0 - ((*region).used_by_allocator + (*region).used_by_user) as f64
                    / (*region).length as f64)
                    * 100.0,
            );
            region = (*region).next;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_checks() {
        let mut allocator = MmapAllocator::new();

        unsafe {
            // Request 1 byte, should call `mmap` with length of PAGE_SIZE.
            let first_addr = allocator.alloc(Layout::new::<u8>()) as *mut u8;
            // We'll use this later to check memory corruption.
            *first_addr = 69;

            // First region should be PAGE_SIZE in length.
            assert_eq!((*allocator.first_region).length, PAGE_SIZE);
            assert_eq!(allocator.num_regions, 1);

            let first_block = (*allocator.first_region).first_block_address();

            // First block should be located after the region header.
            assert_eq!(
                first_block as usize - allocator.first_region as usize,
                mem::size_of::<Region>()
            );

            // Size of first block should be minimum size because we've
            // requested 1 byte only.
            assert_eq!((*first_block).size, MIN_BLOCK_SIZE);

            // First free block should be located after region header.
            assert_eq!(allocator.num_free_blocks, 1);
            assert_eq!(
                (*allocator.first_free_block).size,
                PAGE_SIZE - mem::size_of::<Region>() - mem::size_of::<Block>() * 2 - MIN_BLOCK_SIZE
            );

            // Region should have two blocks, we are using the first one and the
            // other one is free.
            assert_eq!((*allocator.first_region).num_blocks, 2);

            // The remaining free block should be split in two when allocating
            // less size than it can hold.
            let second_addr = allocator.alloc(Layout::array::<u8>(PAGE_SIZE / 2).unwrap());

            for i in 0..(PAGE_SIZE / 2) {
                *(second_addr).add(i) = 69;
            }

            // There are 3 blocks now, last one is still free.
            assert_eq!((*allocator.first_region).num_blocks, 3);
            assert_eq!(allocator.num_free_blocks, 1);

            // Lets try to allocate the entire remaining free block.
            let third_alloc = PAGE_SIZE
                - (*allocator.first_region).used_by_allocator
                - (*allocator.first_region).used_by_user
                - mem::size_of::<Block>();
            let third_addr = allocator.alloc(Layout::array::<u8>(third_alloc).unwrap());

            for i in 0..third_alloc {
                *(third_addr.add(i)) = 69;
            }

            // Number of blocks hasn't changed, but we don't have free blocks
            // anymore.
            assert_eq!((*allocator.first_region).num_blocks, 3);
            assert_eq!(allocator.num_free_blocks, 0);

            // Time for checking memory corruption
            assert_eq!(*first_addr, 69);
            for i in 0..(PAGE_SIZE / 2) {
                assert_eq!((*second_addr.add(i)), 69);
            }
            for i in 0..third_alloc {
                assert_eq!((*third_addr.add(i)), 69);
            }

            // Let's request a bigger chunk so that a new region is used.
            let fourth_alloc = PAGE_SIZE * 2 - PAGE_SIZE / 2;
            let fourth_addr = allocator.alloc(Layout::array::<u8>(fourth_alloc).unwrap());

            assert_eq!(allocator.num_regions, 2);
            assert_eq!(allocator.num_free_blocks, 1);

            // Let's play with dealloc now.
            allocator.dealloc(first_addr);
            assert_eq!((*allocator.first_region).num_blocks, 3);
            assert_eq!(allocator.num_free_blocks, 2);

            allocator.dealloc(third_addr);
            assert_eq!((*allocator.first_region).num_blocks, 3);
            assert_eq!(allocator.num_free_blocks, 3);

            // Now here comes the magic, if we deallocate second addr all blocks
            // in region one should be merged and region should be returned to
            // the kernel.
            allocator.dealloc(second_addr);
            assert_eq!(allocator.num_regions, 1);
            assert_eq!(allocator.num_free_blocks, 1);

            // Same with deallocating fourth alloc
            allocator.dealloc(fourth_addr);
            assert_eq!(allocator.num_regions, 0);
            assert_eq!(allocator.num_free_blocks, 0);
        }
    }
}
