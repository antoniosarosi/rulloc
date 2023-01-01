use std::ptr::NonNull;

use crate::Pointer;

/// Abstraction for platform specific memory handling. The allocator only needs
/// to request pages of memory and return them back when they are no longer in
/// use, but it doesn't care about the APIs offered by the underlying kernel or
/// libraries.
trait PlatformSpecificMemory {
    /// Requests a memory region from the kernel where `length` bytes can be
    /// written safely.
    unsafe fn request_memory(length: usize) -> Pointer<u8>;

    /// Attempts to return `length` bytes starting from `address` to the
    /// underlying kernel. This function will usually be called to discard
    /// entire regions of memory, so length will equal the size of the region.
    unsafe fn return_memory(address: NonNull<u8>, length: usize);

    /// Virtual memory page size in bytes.
    unsafe fn page_size() -> usize;
}

/// Zero sized type that implements [`PlatformSpecificMemory`] for each OS.
pub(crate) struct Platform;

/// Virtual memory page size. 4096 bytes on most computers. This should be a
/// constant but we don't know the value at compile time.
pub(crate) static mut PAGE_SIZE: usize = 0;

/// We only know the value of the page size at runtime by calling into C
/// libraries, so we'll mutate a global variable and reuse it afterwards.
#[inline]
pub(crate) fn page_size() -> usize {
    unsafe {
        if PAGE_SIZE == 0 {
            PAGE_SIZE = Platform::page_size();
        }

        PAGE_SIZE
    }
}

/// Convinience wrapper for [`PlatformSpecificMemory::request_memory`].
#[inline]
pub(crate) unsafe fn request_memory(length: usize) -> Pointer<u8> {
    Platform::request_memory(length)
}

/// Convinience wrapper for [`PlatformSpecificMemory::return_memory`].
#[inline]
pub(crate) unsafe fn return_memory(address: NonNull<u8>, length: usize) {
    Platform::return_memory(address, length)
}

#[cfg(unix)]
#[cfg(not(miri))]
mod unix {
    use std::ptr::{self, NonNull};

    use libc;

    use super::{Platform, PlatformSpecificMemory};
    use crate::Pointer;

    impl PlatformSpecificMemory for Platform {
        unsafe fn request_memory(length: usize) -> Pointer<u8> {
            // Memory protection. Read-Write only.
            let protection = libc::PROT_READ | libc::PROT_WRITE;

            // Memory should be private to our process and not mapped to any file.
            let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;

            // For all the configuration options that `mmap` accepts see
            // https://man7.org/linux/man-pages/man2/mmap.2.html
            match libc::mmap(ptr::null_mut(), length, protection, flags, -1, 0) {
                libc::MAP_FAILED => None,
                address => Some(NonNull::new_unchecked(address).cast()),
            }
        }

        unsafe fn return_memory(address: NonNull<u8>, length: usize) {
            if libc::munmap(address.cast().as_ptr(), length) != 0 {
                // TODO: What should we do here? Panic? Memory region is still
                // valid here, it wasn't unmapped.
            }
        }

        unsafe fn page_size() -> usize {
            libc::sysconf(libc::_SC_PAGE_SIZE) as usize
        }
    }
}

#[cfg(windows)]
#[cfg(not(miri))]
mod windows {
    use std::{mem::MaybeUninit, ptr::NonNull};

    use windows::Win32::System::{Memory, SystemInformation};

    use super::{Platform, PlatformSpecificMemory};
    use crate::Pointer;

    impl PlatformSpecificMemory for Platform {
        unsafe fn request_memory(length: usize) -> Pointer<u8> {
            // Similar to mmap on Linux, Read-Write only.
            let protection = Memory::PAGE_READWRITE;

            // This works a little bit different from mmap, memory has to be
            // reserved first and then committed in order to become usable. We
            // can do both at the same time with one single call.
            let flags = Memory::MEM_RESERVE | Memory::MEM_COMMIT;

            // For more detailed explanations of each parameter, see
            // https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc#parameters
            let address = Memory::VirtualAlloc(None, length, flags, protection);

            NonNull::new(address.cast())
        }

        unsafe fn return_memory(address: NonNull<u8>, _length: usize) {
            // Again, we have to decommit memory first and then release it. We
            // can skip decommitting by specifying length of 0 and MEM_RELEASE
            // flag. See the docs for details:
            // https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualfree#parameters
            let address = address.cast().as_ptr();
            let length = 0;
            let flags = Memory::MEM_RELEASE;

            if !Memory::VirtualFree(address, length, flags).as_bool() {
                // TODO: Release failed, don't know what to do here yet. Same
                // problem as munmap on Linux.
            }
        }

        unsafe fn page_size() -> usize {
            let mut system_info = MaybeUninit::uninit();
            SystemInformation::GetSystemInfo(system_info.as_mut_ptr());

            system_info.assume_init().dwPageSize as usize
        }
    }
}

#[cfg(miri)]
mod miri {
    //! When using Miri, we can't rely on system calls such as `mmap` because
    //! there's no FFI support, so instead we'll use the global allocator to
    //! mock low level memory managament. This is also useful for detecting
    //! memory leaks in our own allocator (regions that are not returned back to
    //! the kernel).

    use std::{alloc, ptr::NonNull};

    use super::{page_size, Platform, PlatformSpecificMemory};
    use crate::Pointer;

    fn to_layout(length: usize) -> alloc::Layout {
        alloc::Layout::from_size_align(length, page_size()).unwrap()
    }

    impl PlatformSpecificMemory for Platform {
        unsafe fn request_memory(length: usize) -> Pointer<u8> {
            NonNull::new(alloc::alloc(to_layout(length)))
        }

        unsafe fn return_memory(address: NonNull<u8>, length: usize) {
            alloc::dealloc(address.as_ptr(), to_layout(length));
        }

        unsafe fn page_size() -> usize {
            4096
        }
    }
}
