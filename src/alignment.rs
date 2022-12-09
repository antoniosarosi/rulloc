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
///  Content addr ->  0x28 |       Padding bytes        |  <---+      <---+
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
/// ammount of padding includes space for the back pointer. The back pointer
/// itself is part of the padding.
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
    //increase `MIN_BLOCK_SIZE`.
    if size < MIN_BLOCK_SIZE {
        size = MIN_BLOCK_SIZE;
    }

    size
}

/// Returns the next address after the given address tha satisfies the required
/// alignment. The received address is never returned because the method
/// described at [`AlignmentBackPointer`] doesn't work if the address is
/// exactly the content address of a block.
///
/// # Implmentation Notes
///
/// This function could by implemented like so:
///
/// ```rust
/// fn basic_impl(address: usize, align: usize) -> usize {
///     let remainder = address % align;
///     address - remainder + align
/// }
///
/// // If we start at 24, next address aligned to 32 is 32.
/// assert_eq!(basic_impl(24, 32), 32);
///
/// // Remember that we never return the address we start at even if it's
/// // already aligned because we need space for the back pointer.
/// assert_eq!(basic_impl(8, 8), 16);
///
/// // More examples.
/// assert_eq!(basic_impl(1008, 8), 1016);
/// assert_eq!(basic_impl(0x18, 8), 0x20);
/// ```
///
/// But since divisions are slow, and in this case unnecessary, we'll do some
/// bit magic instead.
///
/// # Bit Magic
///
/// `align` is guaranteed to be a power of 2, so in binary it will always have a
/// 1 at some bit position while all the rest of bits are 0:
///
/// ```rust
/// assert_eq!(2, 0b0010);
/// assert_eq!(4, 0b0100);
/// assert_eq!(8, 0b1000);
/// ```
///
/// Let's call the bit position that equals 1 `P`. All powers of two have their
/// own P, so an address A is aligned to a power of two if the bits after P in A
/// are equal to 0. Consider the address `0x18`, this address is 8-aligned
/// because 8 in binary is `0000 1000` while `0x18` is `0001 1000`:
///
/// ```text
///                +--- This bit position is P
///                |
///    8 ---> 0000 1000
///                 --- Bits after P
///
/// 0x18 ---> 0001 1000
///                 --- Bits after P are 0 in the address, so 0x18 is aligned
///
/// 0x19 ---> 0001 1001
///                 --- Not all bits are zero, so 0x19 is not 8-aligned
///
/// 0x20 ---> 0010 0000
///                 --- All bits are 0, so 0x20 is 8-aligned
/// ```
///
/// Knowing this, we now have to somehow isolate the bits _after_ the position
/// P in the address so that we can manipulate them to be zero and meet the
/// alignment constraint. To do that, we need a bit mask, which we can obtain
/// by substracting 1 to `align` and applying bitwise NOT over the result:
///
/// ```rust
/// // Let's make it u8 so that we don't have to write 64 bits one by one, but
/// // remember that addresses are the same size as usize.
/// let align: u8 = 8;
/// assert_eq!(8, 0b00001000);
///
/// // Substracting 1 to any power of 2 will make the bits after P equal to 1.
/// let bits_after_P_position = align - 1;
/// assert_eq!(bits_after_P_position, 0b00000111);
///
/// // Now apply bitwise NOT and we get the mask.
/// let bit_mask = !bits_after_P_position;
/// assert_eq!(bit_mask, 0b11111000);
/// ```
///
/// Once we have the bit mask, we can align any address by applying bitwise AND
/// on the address and the mask. For example:
///
/// ```rust
/// let align: u8 = 8;
/// let bits_after_P_position = align - 1;
/// let bit_mask = !bits_after_P_position;
///
///         // Hex                     // Decimal
/// assert_eq!(0x14 & bit_mask, 0x10); // 20 & bit_mask == 16
/// assert_eq!(0x1C & bit_mask, 0x18); // 28 & bit_mask == 24
/// assert_eq!(0x24 & bit_mask, 0x20); // 36 & bit_mask == 32
/// assert_eq!(0x10 & bit_mask, 0x10); // 16 & bit_mask == 16 (stays the same)
/// ```
///
/// Now we know that if we apply this algorithm to any address, it will align
/// the address _downwards_, which is a problem. Consider an alignment of 16:
///
/// ```rust
/// let align: u8 = 16;
/// let bits_after_P_position = align - 1;
/// let bit_mask = !bits_after_P_position;
/// let content_address = 0x18; // Block content starts at 24 in decimal.
///
/// assert_eq!(content_address & bit_mask, 0x10); // Aligns down to 16.
/// ```
///
/// Oops! That shouldn't happen, because we'd return an address that points
/// before the block content, so the user could override the block header or
/// the content of another block if the alignment is really big. To fix this,
/// we are going to offset the original address by `align - 1` bytes and apply
/// the mask to that:
///
/// ```rust
/// let align: u8 = 16;
/// let bits_after_P_position = align - 1;
/// let bit_mask = !bits_after_P_position;
/// let content_address = 0x18; // Block content starts at 24 in decimal.
///
/// // Now this aligns upwards, so its correct.
/// assert_eq!((content_address + align - 1) & bit_mask, 0x20);
/// ```
///
/// In doing so, we prevent the new address from being located before the
/// original address. That's because between `address` and `address + align`
/// there must be another address that meets the required alignment. Refer
/// back to [`AlignmentBackPointer`] for a detailed explanation, but here's how
/// we can conceptualize this in terms of bits and masks:
///
/// ```text
/// +-----------------------------+------+------+-----------+
/// | Variable                    | Dec  | Hex  |    Bin    |
/// +-----------------------------+------+------+-----------+
/// | align                       |  16  | 0x10 | 0001 0000 |
/// | content_address             |  24  | 0x18 | 0001 1000 |
/// | content_address + align - 1 |  39  | 0x27 | 0010 0111 |
/// | bit_mask                    |      |      | 1111 0000 |
/// | result                      |  32  | 0x20 | 0010 0000 |
/// +-----------------------------+------+------+-----------+
/// ```
///
/// Notice how only the bits after P are affected by the mask, while the other
/// bits important for powers of two are not affected. In the case of 39, the
/// fifth bit counting from right to left (starting at 0) stays the same, and
/// 2 ^ 5 = 32, which is aligned to 16. Adding `align - 1` to the address makes
/// sure that the next bit important for alignment is set to 1. Let's see
/// another example:
///
/// ```text
/// +-----------------------------+------+------+-----------+
/// | Variable                    | Dec  | Hex  |    Bin    |
/// +-----------------------------+------+------+-----------+
/// | align                       |  32  | 0x20 | 0010 0000 |
/// | content_address             |  40  | 0x28 | 0010 1000 |
/// | content_address + align - 1 |  71  | 0x47 | 0100 0111 |
/// | bit_mask                    |      |      | 1110 0000 |
/// | result                      |  64  | 0x40 | 0100 0000 |
/// +-----------------------------+------+------+-----------+
/// ```
///
/// Again, the bit responsible for 2 ^ 6 = 64 doesn't change, so we will always
/// get an address that meets the alignment. However, if the content address is
/// already aligned we're going to get back the same address:
///
/// ```rust
/// let align: u8 = 16;
/// let bits_after_P_position = align - 1;
/// let bit_mask = !bits_after_P_position;
/// let content_address = 0x10; // Block content starts at 16 in decimal.
///
/// // Aligned address is equal to the content address.
/// assert_eq!((content_address + align - 1) & bit_mask, 0x10);
/// ```
///
/// This is a problem. We don't want that for our implementation. Let's what's
/// the issue:
///
/// ```text
/// +-----------------------------+------+------+-----------+
/// | Variable                    | Dec  | Hex  |    Bin    |
/// +-----------------------------+------+------+-----------+
/// | align                       |  16  | 0x10 | 0001 0000 |
/// | content_address             |  16  | 0x10 | 0001 1000 |
/// | content_address + align - 1 |  31  | 0x1F | 0001 1111 |
/// | bit_mask                    |      |      | 1111 0000 |
/// | result                      |  16  | 0x10 | 0001 0000 |
/// +-----------------------------+------+------+-----------+
/// ```
///
/// Basically, the bit responsible for 2 ^ 5 = 32 is not set to 1. The alignment
/// is still correct, but remember that we don't want to return the content
/// address to the user because then we have no space to write the back pointer
/// above it as we would override the header struct. So to fix this, we must set
/// to 1 the next bit that raises 2 to its next power. To ensure we do, we'll
/// just get rid of the `- 1`:
///
/// ```rust
/// let align: u8 = 16;
/// let bits_after_P_position = align - 1;
/// let bit_mask = !bits_after_P_position;
/// let content_address = 0x10; // Block content starts at 16 in decimal.
///
/// // Note that there's no -1. Next aligned address excluding the address of
/// // the block content is 0x20, or 32 in decimal.
/// assert_eq!((content_address + align) & bit_mask, 0x20);
/// ```
///
/// This fixes the issue:
///
/// ```text
/// +-------------------------+------+------+-----------+
/// | Variable                | Dec  | Hex  |    Bin    |
/// +-------------------------+------+------+-----------+
/// | align                   |  16  | 0x10 | 0001 0000 |
/// | content_address         |  16  | 0x10 | 0001 1000 |
/// | content_address + align |  32  | 0x20 | 0010 0000 |
/// | bit_mask                |      |      | 1111 0000 |
/// | result                  |  32  | 0x10 | 0010 0000 |
/// +-------------------------+------+------+-----------+
/// ```
///
/// And it still aligns downwards when possible:
///
/// ```text
/// +-------------------------+------+------+-----------+
/// | Variable                | Dec  | Hex  |    Bin    |
/// +-------------------------+------+------+-----------+
/// | align                   |  16  | 0x10 | 0001 0000 |
/// | content_address         |   8  | 0x08 | 0000 1000 |
/// | content_address + align |  24  | 0x18 | 0001 1000 |
/// | bit_mask                |      |      | 1111 0000 |
/// | result                  |  16  | 0x10 | 0001 0000 |
/// +-------------------------+------+------+-----------+
/// ```
///
/// The `-1` is used when we need already aligned addresses to stay the same,
/// but our implementation doesn't need that. I included it here because that's
/// the code you're gonna find if you search for it on the Internet, but we have
/// to tweak it a little bit (pun intented), so we better understand how it
/// works.
///
/// Remember that there are 2 hard problems in computer science: cache
/// invalidation, naming things and off by one errors. And memory allocators
/// might as well make it into that list.
///
/// It's been a long journey, but putting it all together, we get this:
fn next_aligned_address(address: usize, align: usize) -> usize {
    (address + align) & !(align - 1)
}

/// Returns the next address after `address` that meets the alignment
/// constraint. `address` itself is never returned. See [`next_aligned_address`].
pub(crate) unsafe fn next_aligned(address: NonNull<u8>, align: usize) -> NonNull<u8> {
    NonNull::new_unchecked(next_aligned_address(address.as_ptr() as usize, align) as *mut u8)
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

    #[test]
    fn next_aligned_test() {
        // (addr, align, expected addr)
        let params = [
            // 8-aligned
            (4, 8, 8),
            (16, 8, 24),
            (24, 8, 32),
            (8, 16, 16),
            // 16-aligned
            (16, 16, 32),
            (40, 16, 48),
            (48, 16, 64),
            // 32-aligned
            (56, 32, 64),
            (64, 32, 96),
            // Some "real life" cases
            (0x7fc4676a5058, 16, 0x7fc4676a5060),
            (0x7fc4676a5058, 32, 0x7fc4676a5060),
            (0x7fc4676a5058, 64, 0x7fc4676a5080),
        ];

        for (address, align, expected) in params {
            assert_eq!(next_aligned_address(address, align), expected);
        }
    }
}
