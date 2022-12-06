use std::{marker::PhantomData, ptr::NonNull};

use crate::{header::Header, Pointer};

/// Linked list node. See also [`Header<T>`].
pub struct Node<T> {
    pub next: Pointer<Self>,
    pub prev: Pointer<Self>,
    pub data: T,
}

/// Custom linked list implementation for this allocator. This struct was
/// created as an abstraction to reduce duplicated code, isolate some unsafe
/// parts and reduce raw pointer usage. It makes the code harder to follow, so
/// if you want a simpler version without this abstraction check this commit:
/// [`37b7752e2daa6707c93cd7badfa85c168f09aac8`](https://github.com/antoniosarosi/memalloc-rust/blob/37b7752e2daa6707c93cd7badfa85c168f09aac8/src/mmap.rs)
#[derive(Clone, Copy)]
pub struct LinkedList<T> {
    pub head: Pointer<Node<T>>,
    pub tail: Pointer<Node<T>>,
    pub len: usize,
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
        let node = address.as_ptr() as *mut Node<T>;

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
