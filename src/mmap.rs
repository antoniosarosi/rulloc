use std::ptr;

use crate::Pointer;

/// Calls `mmap` and returns the resulting address or `None` if `mmap fails`.
///
/// # Arguments
///
/// * `length` - Length that we should call `mmap` with. This should be a
/// multiple of [`PAGE_SIZE`].
#[cfg(not(miri))]
pub unsafe fn mmap(length: usize) -> Pointer<u8> {
    // C void null pointer. This is what we need to request memory with mmap.
    let null = ptr::null_mut::<libc::c_void>();
    // Memory protection. Read-Write only.
    let protection = libc::PROT_READ | libc::PROT_WRITE;
    // Memory flags. Should be private to our process and not mapped to any
    // file or device (MAP_ANONYMOUS).
    let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

    match libc::mmap(null, length, protection, flags, -1, 0) {
        libc::MAP_FAILED => None,
        address => Some(ptr::NonNull::new_unchecked(address as *mut u8)),
    }
}

/// Calls [`libc::munmap`] on `address` with `length`. `address` must be
/// valid.
#[cfg(not(miri))]
pub unsafe fn munmap(address: *mut u8, length: usize) {
    if libc::munmap(address as *mut libc::c_void, length) != 0 {
        // TODO: What should we do here? Panic? Memory region is still
        // valid here, it wasn't unmapped.
    }
}

#[cfg(miri)]
pub unsafe fn mmap(length: usize) -> Pointer<u8> {
    use std::{alloc::Layout, mem};
    let layout = Layout::array::<u8>(length)
        .unwrap()
        .align_to(mem::size_of::<usize>());
    let address = std::alloc::alloc(layout.unwrap());

    return Some(ptr::NonNull::new_unchecked(address));
}

#[cfg(miri)]
pub unsafe fn munmap(address: *mut u8, length: usize) {
    use std::{alloc::Layout, mem};
    let layout = Layout::array::<u8>(length)
        .unwrap()
        .align_to(mem::size_of::<usize>());
    std::alloc::dealloc(address, layout.unwrap());
}
