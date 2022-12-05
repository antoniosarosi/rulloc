use std::{alloc::Layout, marker::PhantomData, mem, ptr, ptr::NonNull};

use libc;

/// Minimum block size in bytes. Read the documentation of the data structures
/// used for this allocator to understand why it has this value. Specifically,
/// see [`Header<T>`], [`Block`], [`Region`], [`LinkedList<T>`] and especially
/// [`FreeListNode`].
const MIN_BLOCK_SIZE: usize = mem::size_of::<FreeListNode>();

/// Block header size in bytes. See [`Header<T>`] and [`Block`].
const BLOCK_HEADER_SIZE: usize = mem::size_of::<Header<Block>>();

/// Region header size in bytes. See [`Header<T>`] and [`Region`].
const REGION_HEADER_SIZE: usize = mem::size_of::<Header<Region>>();

/// Pointer size in bytes on the current machine.
const POINTER_SIZE: usize = mem::size_of::<usize>();

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

/// Non-null pointer to `T`. We use this in most cases instead of `*mut T`
/// because the compiler will yell at us if we don't write code for the `None`
/// case. I think variance doesn't have much implications here except for
/// [`LinkedList<T>`], but that should probably be covariant anyway.
type Pointer<T> = Option<NonNull<T>>;

/// Linked list node. See also [`Header<T>`].
struct Node<T> {
    next: Pointer<Self>,
    prev: Pointer<Self>,
    data: T,
}

/// Since all the headers we store point to their previous and next header we
/// might as well consider them linked list nodes. This is just a type alias
/// that we use when we want to refer to a block or region header without
/// thinking about linked list nodes.
type Header<T> = Node<T>;

/// Custom linked list implementation for this allocator. This struct was
/// created as an abstraction to reduce duplicated code, isolate some unsafe
/// parts and reduce raw pointer usage. It makes the code harder to follow, so
/// if you want a simpler version without this abstraction check this commit:
/// [`37b7752e2daa6707c93cd7badfa85c168f09aac8`](https://github.com/antoniosarosi/memalloc-rust/blob/37b7752e2daa6707c93cd7badfa85c168f09aac8/src/mmap.rs)
struct LinkedList<T> {
    head: Pointer<Node<T>>,
    tail: Pointer<Node<T>>,
    len: usize,
    marker: PhantomData<T>,
}

/// Memory block specific data. All headers are also linked list nodes, see
/// [`Header<T>`]. In this case, a complete block header would be `Node<Block>`,
/// also known as `Header<Block>`. Here's a graphical representation of how it
/// looks like in memory:
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
struct Block {
    /// Memory region where this block is located.
    region: NonNull<Header<Region>>,
    /// Size of the block excluding `Header<Block>` size.
    size: usize,
    /// Whether this block can be used or not.
    is_free: bool,
}

/// Memory region specific data. All headers are also linked lists nodes, see
/// [`Header<T>`] and [`Block`]. In this case, a complete region header would be
/// `Header<Region>`.
///
/// We use [`libc::mmap`] to request memory regions to the kernel, and we cannot
/// assume that these regions are adjacent because `mmap` might be used outside
/// of this allocator (and that's okay) or we unmapped a previously mapped
/// region, which causes its next and previous regions to be non-adjacent.
/// Therefore, we store regions in a linked list. Each region also contains a
/// linked list of blocks. This is the high level overview:
///
/// ```text
/// +--------+------------------------+      +--------+-------------------------------------+
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// | Region | | Block | -> | Block | | ---> | Region | | Block | -> | Block | -> | Block | |
/// |        | +-------+    +-------+ |      |        | +-------+    +-------+    +-------+ |
/// +--------+------------------------+      ---------+-------------------------------------+
/// ```
struct Region {
    /// Blocks contained within this memory region.
    blocks: LinkedList<Block>,
    /// Size of the region excluding `Header<Region>` size.
    size: usize,
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
/// [`Node<T>`] can only point to instances of itself, so `Node<()>` can only
/// point to `Node<()>`, otherwise [`LinkedList<T>`] implementation won't work
/// for this type. Therefore, the free list doesn't contain pointers to the
/// headers of free blocks, it contains pointers to the *content* of free
/// blocks. If we want to obtain the actual block header given a `Node<()>`, we
/// have to substract [`BLOCK_HEADER_SIZE`] to its address and cast to
/// `Header<Block>` or `Node<Block>`.
///
/// It's not so intuitive because the free list should be `LinkedList<Block>`,
/// but with this implementation it becomes `LinkedList<()>` instead. This,
/// however, allows us to reuse [`LinkedList<T>`] without having to implement
/// traits for different types of nodes to give us their next and previous or
/// having to rely on macros to generate code for that. The only incovinience
/// is that the free list points to content of free blocks instead of their
/// header, but we can easily obtain the actual header.
///
/// I suggest that you check out the following commit, which doesn't have
/// [`LinkedList<T>`], to understand why this method reduces boilerplate:
/// [`37b7752e2daa6707c93cd7badfa85c168f09aac8`](https://github.com/antoniosarosi/memalloc-rust/blob/37b7752e2daa6707c93cd7badfa85c168f09aac8/src/mmap.rs#L649-L687)
type FreeListNode = Node<()>;

/// See [`FreeListNode`].
type FreeList = LinkedList<()>;

/// General purpose allocator. All memory is requested from the kernel using
/// [`libc::mmap`] and some tricks and optimizations are implemented such as
/// free list, block coalescing and block splitting.
pub struct MmapAllocator {
    /// Free list.
    free_blocks: FreeList,
    /// First region that we've allocated with `mmap`.
    regions: LinkedList<Region>,
}

impl<T> Header<T> {
    /// Returns a pointer to a [`Header<T>`] given an address that points right
    /// after a valid [`Header<T>`].
    ///
    /// ```text
    /// +-------------+
    /// |  Header<T>  | <- Returned address points here.
    /// +-------------+
    /// |   Content   | <- Given address should point here.
    /// +-------------+
    /// |     ...     |
    /// +-------------+
    /// |     ...     |
    /// +-------------+
    /// ```
    ///
    /// # Safety
    ///
    /// Caller must guarantee that the given address points exactly to the first
    /// memory cell after a [`Header<T>`]. This function will be mostly used for
    /// deallocating memory, so the allocator user should give us an address
    /// that we previously provided when allocating. As long as that's true,
    /// this is safe, otherwise it's undefined behaviour.
    pub unsafe fn from_content_address(address: NonNull<u8>) -> NonNull<Self> {
        NonNull::new_unchecked(address.as_ptr().sub(mem::size_of::<Self>()) as *mut Self)
    }

    /// Returns the address after the header.
    ///
    /// ```text
    /// +---------+
    /// | Header  | <- Header<T> struct.
    /// +---------+
    /// | Content | <- Returned address points to the first cell after header.
    /// +---------+
    /// |   ...   |
    /// +---------+
    /// |   ...   |
    /// +---------+
    /// ```
    ///
    /// # Safety
    ///
    /// If `header` is a valid `NonNull<Header<T>>`, the offset will return an
    /// address that points right after the header. That address is safe to use
    /// as long as no more than `size` bytes are written, where `size` is a
    /// field of [`Block`] or [`Region`].
    ///
    /// # Notes
    ///
    /// - We are using this function as `Header::content_address_of(header)`
    /// because we want to avoid creating references to `self` to keep Miri
    /// happy. See [Stacked Borrows](https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md).
    pub unsafe fn content_address_of(header: NonNull<Self>) -> *mut u8 {
        header.as_ptr().offset(1) as *mut u8
    }
}

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
        Self::from_content_address(NonNull::new_unchecked(links.as_ptr() as *mut u8))
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

impl Header<Region> {
    /// Returns a pointer to the first block in this region.
    ///
    /// # Safety
    ///
    /// There is **ALWAYS** at least one block in the region.
    pub unsafe fn first_block(&self) -> NonNull<Header<Block>> {
        self.data.blocks.head.unwrap_unchecked()
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
        self.data.blocks.len
    }
}

impl<T> LinkedList<T> {
    /// Creates an empty linked list. No allocations happen because, well, we
    /// are the allocator.
    pub fn new() -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
            marker: PhantomData,
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
    pub unsafe fn append(&mut self, data: T, address: *mut u8) -> NonNull<Header<T>> {
        let node = address as *mut Node<T>;

        *node = Node {
            prev: self.tail,
            next: None,
            data,
        };

        let link = Some(NonNull::new_unchecked(node));

        if let Some(mut tail) = self.tail {
            tail.as_mut().next = link;
        } else {
            self.head = link;
        }

        self.tail = link;
        self.len += 1;

        NonNull::new_unchecked(node)
    }

    /// Inserts a new node with the given `data` right after the given `node`.
    /// New node will be written to `address`, so address must be valid and
    /// non-null.
    ///
    /// # Safety
    ///
    /// Caller must guarantee that both `address` and `node` are valid.
    pub unsafe fn insert_after(
        &mut self,
        mut node: NonNull<Node<T>>,
        data: T,
        address: *mut u8,
    ) -> NonNull<Header<T>> {
        let next = address as *mut Node<T>;

        *next = Node {
            prev: Some(node),
            next: node.as_ref().next,
            data,
        };

        node.as_mut().next = Some(NonNull::new_unchecked(next));

        if node == self.tail.unwrap() {
            self.tail = node.as_ref().next;
        }

        self.len += 1;

        NonNull::new_unchecked(next)
    }

    /// Removes `node` from the linked list. `node` must be valid.
    pub unsafe fn remove(&mut self, mut node: NonNull<Node<T>>) {
        if self.len == 1 {
            self.head = None;
            self.tail = None;
        } else if node == self.head.unwrap() {
            node.as_mut().next.unwrap().as_mut().prev = None;
            self.head = node.as_ref().next;
        } else if node == self.tail.unwrap() {
            node.as_mut().prev.unwrap().as_mut().next = None;
            self.tail = node.as_ref().prev;
        } else {
            let mut next = node.as_ref().next.unwrap();
            let mut prev = node.as_ref().prev.unwrap();
            prev.as_mut().next = Some(next);
            next.as_mut().prev = Some(prev);
        }

        self.len -= 1;
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
            Header::content_address_of(block) as *mut FreeListNode
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

impl MmapAllocator {
    // Constructs a new allocator. No actual allocations happen until memory
    // is requested using [`MmapAllocator::alloc`].
    pub fn new() -> Self {
        Self {
            free_blocks: FreeList::new(),
            regions: LinkedList::new(),
        }
    }

    /// Allocates a new block that can fit at least `layout.size()` bytes.
    /// Because of alignment and headers, it might allocate a bigger block than
    /// needed. As long as no more than `layout.align()` bytes are written on
    /// the content part of the block it should be fine.
    pub unsafe fn alloc(&mut self, mut layout: Layout) -> *mut u8 {
        layout = layout.align_to(POINTER_SIZE).unwrap().pad_to_align();

        let size = if layout.size() >= MIN_BLOCK_SIZE {
            layout.size()
        } else {
            MIN_BLOCK_SIZE
        };

        let free_block = match self.find_free_block(size) {
            Some(block) => block,
            None => {
                let Some(region) = self.request_region(size) else {
                    return ptr::null_mut();
                };
                region.as_ref().first_block()
            }
        };

        self.split_free_block_if_possible(free_block, size);
        self.free_blocks.remove_block(free_block);

        Header::content_address_of(free_block)
    }

    /// Deallocates the given pointer. Memory might not be returned to the OS
    /// if the region where `address` is located still contains used blocks.
    /// However, the freed block will be reused later if possible.
    pub unsafe fn dealloc(&mut self, address: *mut u8) {
        let mut block = Header::<Block>::from_content_address(NonNull::new_unchecked(address));

        // This block is now free as it is about to be deallocated.
        self.free_blocks.append_block(block);

        // If previous block is merged then the address will change.
        block = self.merge_free_blocks_if_possible(block);

        let region = block.as_ref().data.region;

        // All blocks have been merged into one, so we can return this region
        // back to the kernel.
        if region.as_ref().num_blocks() == 1 {
            // The free block in this region is no longer valid because this
            // region is about to be unmapped.
            self.free_blocks.remove_block(block);

            // Region has to be removed before unmapping, otherwise seg fault.
            self.regions.remove(region);

            let length = region.as_ref().total_size() as libc::size_t;
            self.munmap(region.as_ptr() as *mut u8, length);
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
        if total_size < length && total_size + BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE > length {
            length += page_size();
        }

        length
    }

    /// Calls `mmap` and returns the resulting address or `None` if `mmap fails`.
    ///
    /// # Arguments
    ///
    /// * `length` - Length that we should call `mmap` with. This should be a
    /// multiple of [`PAGE_SIZE`].
    unsafe fn mmap(&self, length: usize) -> Pointer<u8> {
        // Simulate mmap call if we are using Miri.
        if cfg!(miri) {
            let layout = Layout::array::<u8>(length)
                .unwrap()
                .align_to(mem::size_of::<usize>());
            let address = std::alloc::alloc(layout.unwrap());

            return Some(NonNull::new_unchecked(address));
        }

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

    /// Calls [`libc::munmap`] on `address` with `length`. `address` must be
    /// valid.
    unsafe fn munmap(&self, address: *mut u8, length: usize) {
        if cfg!(miri) {
            let layout = Layout::array::<u8>(length)
                .unwrap()
                .align_to(mem::size_of::<usize>());
            std::alloc::dealloc(address, layout.unwrap());
        } else {
            if libc::munmap(address as *mut libc::c_void, length) != 0 {
                // TODO: What should we do here? Panic? Memory region is still
                // valid here, it wasn't unmapped.
            }
        }
    }

    /// Requests a new memory region from the kernel where we can fit `size`
    /// bytes plus headers. See [`MmapAllocator::determine_region_length`].
    /// The region is already initiated with a free block that spans across
    /// the entire region.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes (must be aligned to word size) that
    /// need to be allocated **without including any headers**.
    unsafe fn request_region(&mut self, size: usize) -> Pointer<Header<Region>> {
        let length = self.determine_region_length(size);

        let Some(address) = self.mmap(length) else {
            return None;
        };

        let mut region = self.regions.append(
            Region {
                blocks: LinkedList::new(),
                size: length - REGION_HEADER_SIZE,
            },
            address.as_ptr(),
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

        Some(region)
    }

    /// Returns the first free block in the free list or `None` if we didn't
    /// find any.
    unsafe fn find_free_block(&self, size: usize) -> Pointer<Header<Block>> {
        let mut current = self.free_blocks.head;

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
        let address = Header::content_address_of(block).add(size);
        let mut region = block.as_ref().data.region;

        // Append new block next to current block.
        let new_block = region.as_mut().data.blocks.insert_after(
            block,
            Block {
                size: block.as_ref().data.size - size - BLOCK_HEADER_SIZE,
                is_free: true,
                region,
            },
            address,
        );

        self.free_blocks.append_block(new_block);

        // The current block can only hold `size` bytes from now on.
        block.as_mut().data.size = size;
    }

    /// This function performs the inverse of [`Self::split_free_block_if_possible`].
    /// If surrounding blocks are free, then we'll merge them all into one
    /// bigger block.
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
    unsafe fn merge_free_blocks_if_possible(
        &mut self,
        mut block: NonNull<Header<Block>>,
    ) -> NonNull<Header<Block>> {
        let next = block.as_ref().next;
        let prev = block.as_ref().prev;

        if next.is_some() && next.unwrap_unchecked().as_ref().is_free() {
            self.merge_next_block(block);
        }

        if prev.is_some() && prev.unwrap_unchecked().as_ref().is_free() {
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

        // Next block doesn't exist any more.
        block.as_mut().region_mut().data.blocks.remove(next);
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
            let first_region = allocator.regions.head.unwrap();
            assert_eq!(first_region.as_ref().total_size(), PAGE_SIZE);
            assert_eq!(allocator.regions.len, 1);

            let first_block = first_region.as_ref().first_block();

            // First block should be located after the region header.
            assert_eq!(
                first_block.as_ptr() as usize - first_region.as_ptr() as usize,
                REGION_HEADER_SIZE
            );

            // Size of first block should be minimum size because we've
            // requested 1 byte only.
            assert_eq!(first_block.as_ref().size(), MIN_BLOCK_SIZE);
            assert_eq!(allocator.free_blocks.len, 1);

            // Region should have two blocks, we are using the first one and the
            // other one is free.
            assert_eq!(first_region.as_ref().num_blocks(), 2);

            // First free block should match this size because there are
            // two blocks in the region and one of them has minimum size.
            assert_eq!(
                allocator.free_blocks.first_free_block().unwrap().size(),
                PAGE_SIZE - REGION_HEADER_SIZE - BLOCK_HEADER_SIZE * 2 - MIN_BLOCK_SIZE
            );

            // The remaining free block should be split in two when allocating
            // less size than it can hold.
            let second_addr = allocator.alloc(Layout::array::<u8>(PAGE_SIZE / 2).unwrap());

            // We'll check corruption later.
            for i in 0..(PAGE_SIZE / 2) {
                *(second_addr).add(i) = 69;
            }

            // There are 3 blocks now, last one is still free.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(allocator.free_blocks.len, 1);

            // Lets try to allocate the entire remaining free block.
            let third_alloc = PAGE_SIZE
                - REGION_HEADER_SIZE
                - (BLOCK_HEADER_SIZE + MIN_BLOCK_SIZE) // First Alloc
                - (BLOCK_HEADER_SIZE + PAGE_SIZE / 2) // Second Alloc
                - BLOCK_HEADER_SIZE;
            let third_addr = allocator.alloc(Layout::array::<u8>(third_alloc).unwrap());

            for i in 0..third_alloc {
                *(third_addr.add(i)) = 69;
            }

            // Number of blocks hasn't changed, but we don't have free blocks
            // anymore.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(allocator.free_blocks.len, 0);

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

            for i in 0..fourth_alloc {
                *(fourth_addr.add(i)) = 69;
            }

            // We should have a new region and a new free block now.
            assert_eq!(allocator.regions.len, 2);
            assert_eq!(allocator.free_blocks.len, 1);

            // Let's play with dealloc.
            allocator.dealloc(first_addr);

            // After deallocating the first block, we should have a new free
            // block but the number of blocks in the region shouldn't change
            // because no coalescing can happen.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(allocator.free_blocks.len, 2);

            allocator.dealloc(third_addr);

            // Again, after deallocating the third block we should have a new
            // free block but the number of block in the region doesn't change.
            assert_eq!(first_region.as_ref().num_blocks(), 3);
            assert_eq!(allocator.free_blocks.len, 3);

            // Now here comes the magic, if we deallocate second addr all blocks
            // in region one should be merged and region should be returned to
            // the kernel.
            allocator.dealloc(second_addr);
            assert_eq!(allocator.regions.len, 1);
            assert_eq!(allocator.free_blocks.len, 1);

            // Same with deallocating fourth alloc
            allocator.dealloc(fourth_addr);
            assert_eq!(allocator.regions.len, 0);
            assert_eq!(allocator.free_blocks.len, 0);
        }
    }
}
