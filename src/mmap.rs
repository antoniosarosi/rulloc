use std::{alloc::Layout, mem, ptr, ptr::NonNull};

use libc;

use crate::align;

/// Minimum block size in bytes. Read the documentation of the data structures
/// used for this allocator to understand why it has this value.
const MIN_BLOCK_SIZE: usize = mem::size_of::<FreeListNode>();

const BLOCK_HEADER_SIZE: usize = mem::size_of::<Header<Block>>();
const REGION_HEADER_SIZE: usize = mem::size_of::<Header<Region>>();

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

type Link<T> = Option<NonNull<T>>;

struct Node<T> {
    next: Link<Self>,
    prev: Link<Self>,
    data: T,
}

type Header<T> = Node<T>;

impl<T> Header<T> {
    pub unsafe fn from_content_address(address: *mut u8) -> *mut Self {
        address.sub(mem::size_of::<Self>()) as *mut Self
    }

    pub unsafe fn content_address(&self) -> *mut u8 {
        (self as *const Self).offset(1) as *mut u8
    }
}

impl Header<Block> {
    pub unsafe fn from_free_list_node(links: *mut FreeListNode) -> *mut Self {
        Self::from_content_address(links as *mut u8)
    }

    pub unsafe fn free_list_node(&self) -> &FreeListNode {
        let links = self.content_address() as *mut FreeListNode;
        &*links
    }

    pub unsafe fn region(&self) -> &Header<Region> {
        &*self.data.region.as_ptr()
    }

    pub unsafe fn region_mut(&mut self) -> &mut Header<Region> {
        &mut *self.data.region.as_ptr()
    }

    pub fn is_free(&self) -> bool {
        self.data.is_free
    }

    pub fn size(&self) -> usize {
        self.data.size
    }

    pub fn total_size(&self) -> usize {
        BLOCK_HEADER_SIZE + self.data.size
    }
}

impl Header<Region> {
    pub unsafe fn first_block_address(&self) -> *mut u8 {
        self.content_address()
    }

    pub unsafe fn first_block_mut(&self) -> &mut Header<Block> {
        // There is *ALWAYS* at least one block in the region.
        &mut *self.data.blocks.head.unwrap().as_ptr()
    }

    pub fn size(&self) -> usize {
        self.data.size
    }

    pub fn total_size(&self) -> usize {
        REGION_HEADER_SIZE + self.data.size
    }

    pub fn num_blocks(&self) -> usize {
        self.data.blocks.len
    }
}

/// Custom linked list implementation for this allocator.
struct LinkedList<T> {
    head: Link<Node<T>>,
    tail: Link<Node<T>>,
    len: usize,
    marker: std::marker::PhantomData<T>,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
            marker: std::marker::PhantomData,
        }
    }

    /// Appends a new node to the linked list. Since it cannot do allocations
    /// (WE ARE THE ALLOCATOR!) it needs the address where the node should be
    /// written to.
    ///
    /// # SAFETY
    ///
    /// Caller must guarantee that `address` is valid.
    ///
    /// # Arguments
    ///
    /// * `data` - The data that the new node will hold.
    ///
    /// * `address` - Memory address where the new node will be written. Must
    /// be valid and non null.
    pub unsafe fn append(&mut self, data: T, address: *mut u8) -> &mut Header<T> {
        let node = address as *mut Node<T>;

        *node = Node {
            prev: self.tail,
            next: None,
            data,
        };

        let link = Some(NonNull::new_unchecked(node));

        if let Some(tail) = self.tail {
            (*tail.as_ptr()).next = link;
        } else {
            self.head = link;
        }

        self.tail = link;
        self.len += 1;

        &mut *node
    }

    pub unsafe fn insert_after(
        &mut self,
        node: &mut Node<T>,
        data: T,
        address: *mut u8,
    ) -> &mut Header<T> {
        let next = address as *mut Node<T>;

        *next = Node {
            prev: Some(NonNull::new_unchecked(node as *mut Node<T>)),
            next: node.next,
            data,
        };

        node.next = Some(NonNull::new_unchecked(next));

        if node as *mut Node<T> == self.tail.unwrap().as_ptr() {
            self.tail = node.next;
        }

        &mut *next
    }

    pub unsafe fn remove(&mut self, node: &Node<T>) {
        if self.len == 1 {
            self.head = None;
            self.tail = None;
        } else if node as *const Node<T> == self.head.unwrap().as_ptr() {
            (*(*node).next.unwrap().as_ptr()).prev = None;
            self.head = (*node).next;
        } else if node as *const Node<T> == self.tail.unwrap().as_ptr() {
            (*(*node).prev.unwrap().as_ptr()).next = None;
            self.tail = (*node).prev;
        } else {
            let next = &mut *(*node).next.unwrap().as_ptr();
            let prev = &mut *(*node).prev.unwrap().as_ptr();
            prev.next = next.next;
            next.prev = prev.prev;
        }

        self.len += 1;
    }
}

impl LinkedList<()> {
    pub unsafe fn append_block(&mut self, block: &Header<Block>) {
        self.append((), block.content_address());
    }

    pub unsafe fn remove_block(&mut self, block: &Header<Block>) {
        self.remove(block.free_list_node());
    }
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
    region: NonNull<Header<Region>>,
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
    blocks: LinkedList<Block>,
    size: usize,
}

type FreeListNode = Node<()>;

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

/// General   purpose allocator. All memory is requested from the kernel using
/// `mmap` and some tricks and optimizations are implemented such as free list,
/// block coalescing and block splitting.
pub struct MmapAllocator {
    /// Free list.
    free_blocks: LinkedList<()>,
    /// First region that we've allocated with `mmap`.
    regions: LinkedList<Region>,
}

impl MmapAllocator {
    // Constructs a new allocator. No actual allocations happen until memory
    // is requested using [`MmapAllocator::alloc`].
    pub fn new() -> Self {
        Self {
            free_blocks: LinkedList::new(),
            regions: LinkedList::new(),
        }
    }

    /// Allocates a new block that can fit at least `layout.size()` bytes.
    /// Because of alignment and headers, it might allocate a bigger block than
    /// needed. As long as no more than `layout.size()` bytes are written on
    /// the content part of the block it should be fine.
    pub unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let size = align(if layout.size() >= MIN_BLOCK_SIZE {
            layout.size()
        } else {
            MIN_BLOCK_SIZE
        });

        let free_block = match self.find_free_block(size) {
            Some(block) => block,
            None => {
                let Some(region) = self.request_region(size) else {
                    return ptr::null_mut();
                };
                region.first_block_mut()
            }
        };

        self.split_free_block_if_possible(free_block, size);
        self.free_blocks.remove_block(free_block);

        free_block.content_address()
    }

    /// Deallocates the given pointer. Memory might not be returned to the OS
    /// if the region where `address` is located still contains used blocks.
    /// However, the freed block will be reused later if possible.
    pub unsafe fn dealloc(&mut self, address: *mut u8) {
        let mut block = Header::<Block>::from_content_address(address);

        self.free_blocks.append_block(&*block);

        // If left block is merged then the address will change.
        block = self.merge_free_blocks_if_possible(&mut *block);

        let region = *(*block).region();

        // All blocks have been merged into one, so we can return this region
        // back to the kernel.
        if region.num_blocks() == 1 {
            // The free block in this region is no longer valid because this
            // region is about to be unmapped.
            self.free_blocks.remove_block(&*block);
            // Region has to be removed before unmapping, otherwise seg fault.
            self.regions.remove(&*(*block).region());
            let length = region.total_size() as libc::size_t;
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
    unsafe fn mmap(&self, length: usize) -> Option<NonNull<u8>> {
        // C void null pointer. This is what we need to request memory with mmap.
        let null = ptr::null_mut::<libc::c_void>();
        // Memory protection. Read-Write only.
        let protection = libc::PROT_READ | libc::PROT_WRITE;
        // Memory flags. Should be private to our process and not mapped to any
        // file or device (MAP_ANONYMOUS).
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

        match libc::mmap(null, length, protection, flags, -1, 0) {
            libc::MAP_FAILED => None,
            address => Some(NonNull::new_unchecked(address as *mut u8)),
        }
    }

    /// Requests a new memory region from the kernel where we can fit `size`
    /// bytes plus headers. See [`MmapAllocator::determine_region_length`].
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes (must be aligned to word size) that
    /// need to be allocated without including any headers.
    unsafe fn request_region(&mut self, size: usize) -> Option<&mut Header<Region>> {
        let length = self.determine_region_length(size);
        let Some(address) = self.mmap(length) else { return None };

        let region = self.regions.append(
            Region {
                blocks: LinkedList::new(),
                size: length,
            },
            address.as_ptr(),
        );

        let block = region.data.blocks.append(
            Block {
                size: region.size() - BLOCK_HEADER_SIZE,
                is_free: true,
                region: NonNull::new_unchecked(region),
            },
            region.first_block_address(),
        );

        self.free_blocks.append_block(block);

        Some(region)
    }

    /// Returns the first free block in the free list or null if none was found.
    unsafe fn find_free_block(&self, size: usize) -> Option<&mut Header<Block>> {
        let mut current = self.free_blocks.head;

        while let Some(node) = current {
            let block = Header::<Block>::from_free_list_node(node.as_ptr());

            if (*block).data.size >= size {
                return Some(&mut *block);
            }

            current = (*node.as_ptr()).next;
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
    unsafe fn split_free_block_if_possible(&mut self, block: &mut Header<Block>, size: usize) {
        // If there's not enough space available we can't split the block.
        if block.data.size < size + BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE {
            return;
        }

        // If we can, the block is located `size` bytes after the initial
        // content address of the current block.
        let address = block.content_address().add(size);
        let region = &mut *block.data.region.as_ptr();
        let new_block = Block {
            size: block.data.size - size - BLOCK_HEADER_SIZE,
            is_free: true,
            region: NonNull::new_unchecked(region),
        };

        // Append new block next to current block.
        let new_block = region.data.blocks.insert_after(block, new_block, address);

        // Append new block to free list.
        self.free_blocks.append_block(new_block);

        // The current block can only hold `size` bytes from now on.
        block.data.size = size;
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
    unsafe fn merge_free_blocks_if_possible(
        &mut self,
        mut block: &mut Header<Block>,
    ) -> &mut Header<Block> {
        if block.next.is_some() && (*block.next.unwrap().as_ptr()).is_free() {
            self.merge_next_block(block);
        }

        if block.prev.is_some() && (*block.prev.unwrap().as_ptr()).is_free() {
            block = &mut (*block.prev.unwrap().as_ptr());
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
    unsafe fn merge_next_block(&mut self, block: &mut Header<Block>) {
        // First update free list. The new bigger block will become the
        // last block, and the 2 old smaller blocks will "dissapear" from the
        // list.
        let next = &mut *block.next.unwrap().as_ptr();
        self.free_blocks.remove_block(next);
        self.free_blocks.remove_block(block);
        self.free_blocks.append_block(block);

        // Now this block is bigger.
        block.data.size += next.total_size();

        block.region_mut().data.blocks.remove(next);
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
