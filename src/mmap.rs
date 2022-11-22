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
struct Region {
    /// Total size of the region, this is the length we called `mmap` with. So
    /// it includes everything, from headers to content.
    length: usize,
    /// Amount of bytes in this region used by the allocator.
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
pub struct MmapAllocator {
    /// First available block. Could be located in any region.
    first_free_block: *mut Block,
    /// Last available block. Could be located in any region.
    last_free_block: *mut Block,
    /// First region that we've allocated with `mmap`.
    first_region: *mut Region,
    /// Last region allocated with `mmap`.
    last_region: *mut Region,
    /// Number of memory regions in the linked list.
    num_regions: usize,
}

impl Block {
    /// Returns the block header associated to the given `address`.
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
        let links = self.content_address() as *mut FreeBlockLinks;
        (*links).next
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
        let links = self.content_address() as *mut FreeBlockLinks;
        (*links).prev
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

        if !free_block.is_null() {
            // TODO: Block splitting.
            (*free_block).is_free = false;
            (*(*free_block).region).used_by_user += content_size;
            (*(*free_block).region).used_by_allocator -= mem::size_of::<FreeBlockLinks>();
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
        (*region).used_by_user += content_size;
        (*region).num_blocks += 1;

        // We've allocated more than needed, so now we have a free block.
        if (*region).length > mem::size_of::<Region>() + mem::size_of::<Block>() + content_size {
            // Remaining chunk size.
            let chunk_size = (*region).length - mem::size_of::<Region>() - (*block).total_size();

            // Write block.
            free_block = (*block).content_address().add(content_size) as *mut Block;
            *free_block = Block {
                region,
                is_free: true,
                next: ptr::null_mut(),
                prev: block,
                size: chunk_size - mem::size_of::<Block>(),
            };

            // Write links to previous and next free blocks.
            let content_address = (*free_block).content_address() as *mut FreeBlockLinks;
            *content_address = FreeBlockLinks {
                next: ptr::null_mut(),
                prev: self.last_free_block,
            };

            // Update stats.
            (*region).num_blocks += 1;
            (*region).used_by_allocator += mem::size_of::<Block>();

            // Update free list.
            if self.first_free_block.is_null() {
                self.first_free_block = free_block;
                self.last_free_block = free_block;
            } else {
                (*self.last_free_block).next = free_block;
                self.last_free_block = free_block;
            }
        }

        (*block).content_address()
    }

    pub unsafe fn dealloc(&self, address: *mut u8) {
        let mut block = Block::from_content_address(address);

        // TODO: Block coalescing.

        (*block).is_free = true;

        (*(*block).region).used_by_allocator += mem::size_of::<FreeBlockLinks>();
        (*(*block).region).used_by_user -= (*block).size;

        let length = (*(*block).region).length as libc::size_t;

        if libc::munmap(address as *mut libc::c_void, length) != 0 {
            // TODO: What should we do here? Panic?
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
        // block which later might be split into multiple blocks after
        // deallocations.
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
            self.last_region = region;
        } else {
            (*self.last_region).next = region;
            self.last_region = region;
        }

        self.num_regions += 1;

        region
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc() {
        let mut allocator = MmapAllocator::new();

        unsafe {
            let first_addr =
                allocator.alloc(Layout::array::<u8>(4096 - 88 - 48).unwrap()) as *mut u8;
            *first_addr = 100;

            assert_eq!(*first_addr, 100);
        }
    }
}
