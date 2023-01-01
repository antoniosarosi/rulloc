use std::{marker::PhantomData, ptr::NonNull};

use crate::{header::Header, Pointer};

/// Linked list node. See also [`Header<T>`].
pub(crate) struct Node<T> {
    pub next: Pointer<Self>,
    pub prev: Pointer<Self>,
    pub data: T,
}

/// Custom linked list implementation for this allocator. Each
/// [`crate::bucket::Bucket`] has to manage 3 linked list structures: list of
/// regions (see [`crate::region::Region`]), list of blocks inside every region
/// (see [`crate::block::Block`]) and list of free blocks (see
/// [`crate::freelist`]). This struct is reused for all mentioned cases.
pub(crate) struct LinkedList<T> {
    head: Pointer<Node<T>>,
    tail: Pointer<Node<T>>,
    len: usize,
    marker: PhantomData<T>,
}

/// Low level iterator for the linked list.
pub(crate) struct Iter<T> {
    current: Pointer<Node<T>>,
    len: usize,
    marker: PhantomData<T>,
}

impl<T> LinkedList<T> {
    /// Creates an empty linked list. No allocations happen because, well, we
    /// are the allocator.
    pub const fn new() -> Self {
        Self {
            head: None,
            tail: None,
            len: 0,
            marker: PhantomData,
        }
    }

    /// Number of elements in the list.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// First element in the list.
    #[inline]
    pub fn first(&self) -> Pointer<Header<T>> {
        self.head
    }

    /// Last element in the list. For now it's only used in tests.
    #[cfg(test)]
    #[inline]
    pub fn last(&self) -> Pointer<Header<T>> {
        self.tail
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
    pub unsafe fn append(&mut self, data: T, address: NonNull<u8>) -> NonNull<Header<T>> {
        let node = address.cast::<Node<T>>();

        node.as_ptr().write(Node {
            prev: self.tail,
            next: None,
            data,
        });

        if let Some(mut tail) = self.tail {
            tail.as_mut().next = Some(node);
        } else {
            self.head = Some(node);
        }

        self.tail = Some(node);
        self.len += 1;

        node
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
        address: NonNull<u8>,
    ) -> NonNull<Header<T>> {
        let new_node = address.cast::<Node<T>>();

        new_node.as_ptr().write(Node {
            prev: Some(node),
            next: node.as_ref().next,
            data,
        });

        if node == self.tail.unwrap() {
            self.tail = Some(new_node);
        } else {
            node.as_ref().next.unwrap().as_mut().prev = Some(new_node);
        }

        node.as_mut().next = Some(new_node);

        self.len += 1;

        new_node
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

    /// Returns an iterator over the linked list elements. The values are
    /// [`NonNull<Header<T>>`], we don't want to deal with moves and references
    /// in this collection. This collection will never be dropped, the allocator
    /// just calls [`libc::munmap`] to return memory regions back to the kernel.
    /// See [`Drop`] implementation for [`crate::bucket::Bucket`].
    pub fn iter(&self) -> Iter<T> {
        Iter {
            current: self.head,
            len: self.len,
            marker: PhantomData,
        }
    }
}

impl<T> Iterator for Iter<T> {
    type Item = NonNull<Node<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.current.map(|node| unsafe {
            self.current = node.as_ref().next;
            self.len -= 1;
            node
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<T> IntoIterator for &LinkedList<T> {
    type Item = NonNull<Node<T>>;

    type IntoIter = Iter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use std::{mem, ptr::NonNull};

    use super::*;
    use crate::platform;

    #[test]
    fn linked_list_operations() {
        unsafe {
            let mut list: LinkedList<u8> = LinkedList::new();
            let region = platform::request_memory(platform::page_size()).unwrap();
            let size = mem::size_of::<Node<u8>>();

            // N1 <-> N2 <-> N3
            let node1 = list.append(1, region);
            let node2 = list.append(2, NonNull::new_unchecked(region.as_ptr().add(size)));
            let node3 = list.append(3, NonNull::new_unchecked(region.as_ptr().add(size * 2)));

            assert_eq!(list.len, 3);

            assert_eq!(node1.as_ref().data, 1);
            assert_eq!(node2.as_ref().data, 2);
            assert_eq!(node3.as_ref().data, 3);

            assert_eq!(list.head, Some(node1));
            assert_eq!(list.tail, Some(node3));

            assert_eq!(node1.as_ref().next, Some(node2));
            assert_eq!(node1.as_ref().prev, None);

            assert_eq!(node2.as_ref().next, Some(node3));
            assert_eq!(node2.as_ref().prev, Some(node1));

            assert_eq!(node3.as_ref().next, None);
            assert_eq!(node3.as_ref().prev, Some(node2));

            // N1 <-> N2 <-> N4 <-> N3
            let node4 = list.insert_after(
                node2,
                4,
                NonNull::new_unchecked(region.as_ptr().add(size * 3)),
            );

            assert_eq!(list.len, 4);

            assert_eq!(list.tail, Some(node3));

            assert_eq!(node4.as_ref().data, 4);
            assert_eq!(node4.as_ref().next, Some(node3));
            assert_eq!(node4.as_ref().prev, Some(node2));

            assert_eq!(node2.as_ref().next, Some(node4));
            assert_eq!(node2.as_ref().prev, Some(node1));

            assert_eq!(node3.as_ref().next, None);
            assert_eq!(node3.as_ref().prev, Some(node4));

            // N1 <-> N2 <-> N3
            list.remove(node4);

            assert_eq!(list.len, 3);

            assert_eq!(node2.as_ref().next, Some(node3));
            assert_eq!(node2.as_ref().prev, Some(node1));

            assert_eq!(node3.as_ref().next, None);
            assert_eq!(node3.as_ref().prev, Some(node2));

            // N1 <-> N2
            list.remove(node3);

            assert_eq!(list.len, 2);

            assert_eq!(Some(node1), list.head);
            assert_eq!(Some(node2), list.tail);
            assert_eq!(node2.as_ref().next, None);
            assert_eq!(node2.as_ref().prev, Some(node1));

            // N2
            list.remove(node1);

            assert_eq!(list.len, 1);

            assert_eq!(Some(node2), list.head);
            assert_eq!(Some(node2), list.tail);
            assert_eq!(node2.as_ref().next, None);
            assert_eq!(node2.as_ref().prev, None);

            // Empty
            list.remove(node2);
            assert_eq!(list.tail, None);
            assert_eq!(list.head, None);
            assert_eq!(list.len, 0);

            platform::return_memory(region, platform::page_size());
        }
    }
}
