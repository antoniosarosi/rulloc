//! Support for strict memory alignment constraints. By implementing
//! [`std::alloc::Allocator`], our allocator must meet the alignment constraints
//! provided by the caller or return an error if it doesn't. See
//! [`AlignmentBackPointer`] to understand how we deal with this.

use std::{alloc::Layout, mem, ptr::NonNull};

use crate::{
    block::{Block, MIN_BLOCK_SIZE},
    header::Header,
};

/// Pointer size in bytes on the current machine (or target architecture). Most
/// personal computers nowadays are 64 bit machines, so this is going to equal
/// 8 in most cases.
pub(crate) const POINTER_SIZE: usize = mem::size_of::<usize>();

/// This is needed to satisfy alignment constraints. Quick reminder of how
/// memory alignment works:
///
/// ```text
///      +-------------+
/// 0x00 | First byte  |
///      +-------------+
/// 0x01 | Second byte |
///      +-------------+
/// 0x02 | Third byte  |
///      +-------------+
/// 0x03 |     ...     |
///      +-------------+
/// 0x04 |     ...     |
///      +-------------+
/// ...  |     ...     |
///      +-------------+
/// 0x0F |     ...     |
///      +-------------+
/// ```
///
/// Consider the figure above. If we receive an allocation request of alignment
/// 1, we can return any pointer in the range `0x00..=0x0F`, because all
/// these values are evenly divisible by 1. However, if we receive an
/// allocation request of alignment 2, we can only return the pointers in
/// `(0x00..=0x0F).filter(|p| p % 2 == 0)`, or in other words only addresses
/// that are evenly divisible by 2. That's basically memory alignment as far
/// as our allocator is concerned.
///
/// Alignment can also be described in terms of multiples: the address `0x04` is
/// 2-aligned because 4 is a multiple of 2. Or more generally, an address A is
/// N-aligned if A is a multiple of N. Note that N must be a power of 2.
///
/// Alignment is important because aligned accesses are optimized at the
/// hardware level, and some processors don't even allow unaligned memory
/// accesses. Since our allocator implements [`std::alloc::Allocator`], it has
/// to satisfy the alignment requirements provided by the user.
///
/// Before we dive into how exactly are we going to satisfy alignments, let's
/// first review how our block headers relate to this. All our structs are
/// aligned to [`mem::size_of::<usize>()`] bytes because their largest field is
/// always [`usize`] or [`NonNull`], which is the same size as [`usize`]. So
/// this should always be true:
///
/// [`mem::align_of::<Header<Block>>()`] == [`mem::size_of::<usize>()`]
///
/// From now on, let's suppose that [`mem::size_of::<usize>()`] equals 8 for
/// simplicity, that's true on most computers these days anyway. So now the
/// statement above becomes:
///
/// [`mem::align_of::<Header<Block>>()`] == 8
///
/// That means that if we write a [`Header<Block>`] struct at an address that is
/// also aligned to 8, we can guarantee that the content address of the block is
/// aligned to 8 as well:
///
/// ```text
///                         +----------------------------+
/// Header address  -> 0x00 | pointer to next block      |  <---+
///                         +----------------------------+      |
///                    0x08 | pointer to prev block      |      |
///                         +----------------------------+      |
///                    0x10 | pointer to block region    |      |
///                         +----------------------------+      | <Header<Block>>
///                    0x18 | block size                 |      |
///                         +----------------------------+      |
///                    0x20 | is free flag (1 byte)      |      |
///                         +----------------------------+      |
///                    0x21 | padding (struct alignment) |  <---+
///                         +----------------------------+
/// Content address -> 0x28 |       Block content        |  <---+
///                         |            ...             |      |
///                         |            ...             |      | Addressable content
///                         |            ...             |      |
///                         |            ...             |  <---+
///                         +----------------------------+
/// ```
///
/// The address `0x28` (40 in decimal) is 8-aligned. If we simply returned that
/// address to the caller, we would cover alignment requirements of 1, 2, 4 and
/// 8. That's because all multiples of 8 are also multiples of 1, 2 and 4. So
/// that's exactly what we are going to do whenever the alignment requirement is
/// 1, 2, 4 or 8.
///
/// But the problem is that this mathematical property doesn't go backwards, so
/// if the user has an alignment requirement of 16, we cannot return an
/// 8-aligned address because not all multiples of 8 are multiples of 16.
///
/// In order to address this issue (pun intended), we are going to store a
/// controlled padding at the beginning of the block content and place a pointer
/// to the block header right above the address that we give to the user. We'll
/// call this pointer "back pointer" from now on, because it points back to the
/// header. Let's first analyze an example of a 32-aligned request, because it's
/// easier to understand than a 16-aligned request:
///
/// ```text
///                        +----------------------------+
/// Header address -> 0x00 | pointer to next block      |  <---+
///                        +----------------------------+      |
///                   0x08 | pointer to prev block      |      |
///                        +----------------------------+      |
///                   0x10 | pointer to block region    |      |
///                        +----------------------------+      | <Header<Block>>
///                   0x18 | block size                 |      |
///                        +----------------------------+      |
///                   0x20 | is free flag (1 byte)      |      |
///                        +----------------------------+      |
///                   0x21 | padding (struct alignment) |  <---+
///                        +----------------------------+
///   Content addr -> 0x28 |       Padding bytes        |  <---+      <---+
///                        +----------------------------+      |          |
///                   0x30 |       Padding bytes        |      | Padding  |
///                        +----------------------------+      |          |
///   Back pointer -> 0x38 | 0x00 (Header struct addr)  |  <---+          |
///                        +----------------------------+                 | Addressable content
///   User pointer -> 0x40 |        User content        |                 |
///                        +----------------------------+                 |
///                        |            ...             |                 |
///                        |            ...             |                 |
///                        |            ...             |             <---+
///                        +----------------------------+
/// ```
///
/// The block content address, in this case, starts at `0x28` (40 in decimal),
/// but the only guarantee we have about this address is that it is 8-aligned,
/// nothing else. It might also be 16-aligned or 32-aligned, because _some_
/// multiples of 8 are also multiples of 16 and 32, but not all of them. So the
/// only thing we know for sure is that the content address is 8-aligned, we
/// cannot rely on anything else.
///
/// Therefore, what we are going to do is find a block that can fit the size
/// requested by the user plus the requested alignment. If the user needs a
/// chunk of 96 bytes aligned to 32, we are going to make sure that our block
/// can fit 96 + 32 bytes. That way, we are certain that we can introduce
/// enough padding at the beginning of the block content so that we return an
/// address that's 32-aligned and can also fit 96 bytes.
///
/// Going back to the figure above, suppose we found (or created) such block and
/// its header address is `0x00`, while its content address is `0x28`. The
/// content address in decimal is 40, so to make it 32-aligned we need to add
/// 24 bytes, which gives us the address 64 or `0x40` in hex. This address is
/// 32-aligned and can fit 96 bytes for sure because we've only added 24 bytes
/// of padding and the block can fit 96 + 32 in total. Also, note that the
/// amount of padding includes space for the back pointer. The back pointer
/// itself is part of the padding. That's why we've showed a 32-aligned example
/// first because this is how 16-aligned looks like given the same block
/// address:
///
/// ```text
///                        +----------------------------+
/// Header address -> 0x00 | pointer to next block      |  <---+
///                        +----------------------------+      |
///                   0x08 | pointer to prev block      |      |
///                        +----------------------------+      |
///                   0x10 | pointer to block region    |      |
///                        +----------------------------+      | <Header<Block>>
///                   0x18 | block size                 |      |
///                        +----------------------------+      |
///                   0x20 | is free flag (1 byte)      |      |
///                        +----------------------------+      |
///                   0x21 | padding (struct alignment) |  <---+
///                        +----------------------------+
///   Back pointer -> 0x28 | 0x00 (Header struct addr)  |  <--- This address is
///                        +----------------------------+       the content
///   User pointer -> 0x30 |        User content        |       address, padding
///                        +----------------------------+       and back pointer
///                        |            ...             |       at the same time
///                        |            ...             |
///                        |            ...             |
///                        +----------------------------+
/// ```
///
/// Notice how `0x28` (40 in decimal) stores 8 bytes of padding and also the
/// back pointer. The address `0x30` (48 in decimal) is the first address
/// aligned to 16, and that should be the user poiner.
///
/// # Padding
///
/// The exact padding that we need to add to an 8-aligned address to obtain a
/// new address that meets the requirements depends on the requested alignment.
/// If the requested alignment is R and R is a power of 2 greater than 8, given
/// any 8-aligned address A we have to add `8 * P` bytes to A to obtain a new
/// R-aligned address, where P is a number in the range `1..=(R / 8)`.
///
/// Let's apply this formula to the 32 alignment example in question:
/// We start at the address 40 in decimal, and we add 8 * 3 bytes to obtain the
/// address 64, which is 32-aligned. If we started at the address 48 in decimal,
/// we'd have to add 8 * 2 bytes to reach the address 64. If we started at the
/// address 56 in decimal, we'd have to add 8 * 1 bytes to reach 64.
///
/// However, if we started exactly at the address 64, we'd have to add 8 * 4
/// bytes to reach the address 96, which is 32 aligned. We are not going to
/// return the address 64 because there's no space above it, we cannot place the
/// back pointer above it as this would override some field of the header
/// struct. This makes the padding in the worst case scenario equal to the
/// required alignment.
///
/// Moreover, we need to make a distinction between blocks that contain a back
/// pointer and blocks that don't, because callers will only give us their own
/// pointers when deallocating. In order to do that, we'll take advantage of the
/// fact that [`std::alloc::Allocator::deallocate`] also forces the caller to
/// give us the layout that was used to allocate a pointer. With this usefull
/// information, we know the exact alignment and we can suppose that all
/// alignments greater than 8 contain a back pointer. Otherwise, suppose that
/// the pointer we are deallocating points exactly at the content address of the
/// block, no padding, so to obtain the block header simply substract the size
/// of it. That's the algorithm.
///
/// If for some reason the API changes and the layout is no longer required on
/// deallocations, we'll have to force all blocks to contain a backpointer,
/// which we can do by introducing 8 bytes of padding even if the alignment is
/// 8. We'll use the padding to simply store the back pointer. Alignments
/// greater than 8 would work the same as described above. As long as the layout
/// is required on deallocation we are not going to store unnecessary back
/// pointers because most allocations will need alignments of 8 or less anyway.
///
/// Last but not least, this method should work on 32 bit machines, the
/// explanation is the same as above but reducing 8 to 4 and dividing all
/// addresses by 2.
pub(crate) type AlignmentBackPointer = NonNull<Header<Block>>;

/// Returns a pointer to the [`AlignmentBackPointer`] of the given `address`.
#[inline]
pub(crate) unsafe fn back_pointer_of(address: NonNull<u8>) -> NonNull<AlignmentBackPointer> {
    NonNull::new_unchecked(address.cast::<AlignmentBackPointer>().as_ptr().offset(-1))
}

/// Returns the minimum block size that allows the methods described at
/// [`AlignmentBackPointer`] to be successfully implemented.
pub(crate) fn minimum_block_size_needed_for(layout: Layout) -> usize {
    // Make sure that the next address is aligned to at least pointer size,
    // because internally we need all header addresses to be aligned to
    // pointer size.
    let mut size = layout.size() + layout.padding_needed_for(POINTER_SIZE);

    // Now if the layout alignment is greater than pointer size, add extra
    // space to make sure we can introduce enough padding to meet the alignment.
    if layout.align() > POINTER_SIZE {
        size += layout.align();
    }

    // If after all of the above for some reason the size is still less than the
    // required minimum, then just set it to the minimum. This can only happen
    // if `layout.align() <= POINTER_SIZE`, because otherwise the if statement
    // above would have increased the size by at least `2 * POINTER_SIZE`. Note
    // that `2 * POINTER_SIZE == MIN_BLOCK_SIZE` because we use free blocks to
    // store 2 pointers for the freelist. So this could actually be an else
    // statement instead, but we'll leave it as an if statement in case we
    // increase `MIN_BLOCK_SIZE`.
    if size < MIN_BLOCK_SIZE {
        size = MIN_BLOCK_SIZE;
    }

    size
}

/// Returns the minimum block size needed to allocate `layout` without taking
/// `layout.align()` into consideration. This is useful for reallocations,
/// where an address might be already aligned so no need to account for
/// padding.
pub(crate) fn minimum_block_size_excluding_padding(layout: Layout) -> usize {
    if layout.size() <= MIN_BLOCK_SIZE {
        MIN_BLOCK_SIZE // This size is already aligned to POINTER_SIZE
    } else {
        layout.size() + layout.padding_needed_for(POINTER_SIZE)
    }
}

/// Returns the next address after the given address tha satisfies the required
/// alignment as well as the padding or offset needed. The given address is
/// **ONLY** returned if `align == POINTER_SIZE` because otherwise the method
/// described at [`AlignmentBackPointer`] cannot be implemented.
pub(crate) unsafe fn next_aligned(address: NonNull<u8>, align: usize) -> (NonNull<u8>, usize) {
    let align_offset = address.as_ptr().align_offset(align);

    // Remember that we don't have space for the back pointer if we don't
    // add at least one `POINTER_SIZE` of padding, so the if statement would
    // be the worst case scenario where we have to add the entire alignment as
    // padding. Otherwise add whatever we've computed above. Note that if
    // `align == POINTER_SIZE` then padding should always be 0 because we work
    // with pointer size aligned addresses internally.
    let padding = if align > POINTER_SIZE && align_offset == 0 {
        align
    } else {
        align_offset
    };

    let next_aligned = address.as_ptr().map_addr(|addr| addr + padding);

    (NonNull::new_unchecked(next_aligned), padding)
}

/// See [`next_aligned`].
#[inline]
pub(crate) unsafe fn padding_needed_to_align(address: NonNull<u8>, align: usize) -> usize {
    let (_, padding) = next_aligned(address, align);
    padding
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_block_size() {
        // (size, align, expected size)
        let layouts = [
            (1, 2, MIN_BLOCK_SIZE),
            (1, 4, MIN_BLOCK_SIZE),
            (1, 8, MIN_BLOCK_SIZE),
            (1, 16, 16 + POINTER_SIZE),
            (8, 16, 16 + 8),
            (14, 16, 16 + 16),
            (16, 16, 16 + 16),
            (96, 16, 16 + 96),
            (1, 32, 32 + POINTER_SIZE),
            (13, 32, 32 + 16),
            (1024, 32, 32 + 1024),
            (512, 64, 64 + 512),
            (510, 64, 64 + 512),
            (4096, 4096, 4096 + 4096),
            (1, 4096, 4096 + POINTER_SIZE),
        ];

        for (size, align, expected) in layouts {
            let layout = Layout::from_size_align(size, align).unwrap();
            assert_eq!(minimum_block_size_needed_for(layout), expected);
        }
    }
}
