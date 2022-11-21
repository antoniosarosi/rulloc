use std::mem;

/// Returns the machine word alignment for the given size.
///
/// # Examples
///
/// ```rust
/// use std::mem;
/// use memalloc::align;
///
/// match mem::size_of::<usize>() {
///     8 => assert_eq!(align(13), 16), // 64 bit machine.
///     4 => assert_eq!(align(11), 12), // 32 bit machine.
///     _ => {}, // some other power of two, will work the same.
/// };
/// ```
#[inline]
pub fn align(to_be_aligned: usize) -> usize {
    (to_be_aligned + mem::size_of::<usize>() - 1) & !(mem::size_of::<usize>() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align() {
        let ptr_size = mem::size_of::<usize>();

        let mut alignments = Vec::new();

        for i in 0..10 {
            // On 64 bit machine: (1..8), (9..16), (16..24) and so on.
            let sizes = (ptr_size * i + 1)..=(ptr_size * (i + 1));
            // Matching the sizes above, this would be: 8, 16, 24 and so on.
            let expected_alignment = ptr_size * (i + 1);
            alignments.push((sizes, expected_alignment));
        }

        for (sizes, expected) in alignments {
            for size in sizes {
                assert_eq!(expected, align(size));
            }
        }
    }
}
