use std::ptr;

use crate::Pointer;

/// Calls [`libc::mmap`] and returns the resulting address or `None` if `mmap`
/// fails.
///
/// # Arguments
///
/// * `length` - Length that we should call `mmap` with. This should be a
/// multiple of [`crate::region::PAGE_SIZE`]. See
/// [`crate::region::determine_region_length`].
#[cfg(not(miri))]
pub(crate) unsafe fn mmap(length: usize) -> Pointer<u8> {
    // C void null pointer. This is what we need to request memory with mmap.
    let null = ptr::null_mut::<libc::c_void>();

    // Memory protection. Read-Write only.
    let protection = libc::PROT_READ | libc::PROT_WRITE;

    // Memory should be private to our process and not mapped to any file.
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

    // For all the configuration options that `mmap` accepts see
    // https://man7.org/linux/man-pages/man2/mmap.2.html
    match libc::mmap(null, length, protection, flags, -1, 0) {
        libc::MAP_FAILED => None,
        address => Some(ptr::NonNull::new_unchecked(address as *mut u8)),
    }
}

/// Calls [`libc::munmap`] on `address` with `length`. `address` must be
/// valid.
#[cfg(not(miri))]
pub(crate) unsafe fn munmap(address: ptr::NonNull<u8>, length: usize) {
    if libc::munmap(address.as_ptr() as *mut libc::c_void, length) != 0 {
        // TODO: What should we do here? Panic? Memory region is still
        // valid here, it wasn't unmapped.
    }
}

/// Maps a given `length` in bytes to a layout. Used only for Miri tests.
#[cfg(miri)]
fn to_layout(length: usize) -> std::alloc::Layout {
    std::alloc::Layout::array::<u8>(length)
        .unwrap()
        .align_to(std::mem::size_of::<usize>())
        .unwrap()
}

/// When running on Miri, all `mmap` calls are mocked using the global
/// allocator. If we are the global allocator, this won't work. See
/// `examples/global.rs` for an explanation of why.
#[cfg(miri)]
pub(crate) unsafe fn mmap(length: usize) -> Pointer<u8> {
    let address = std::alloc::alloc(to_layout(length));

    Some(ptr::NonNull::new_unchecked(address))
}

/// Same as before, simulate `munmap` calls using the global allocator.
#[cfg(miri)]
pub(crate) unsafe fn munmap(address: ptr::NonNull<u8>, length: usize) {
    std::alloc::dealloc(address.as_ptr(), to_layout(length));
}
