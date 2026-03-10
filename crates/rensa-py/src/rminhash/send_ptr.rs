#[derive(Clone, Copy)]
pub(in crate::rminhash) struct SendPtr<T> {
  ptr: *mut T,
}

// SAFETY: `SendPtr` is only constructed for pointers into Rust-owned allocations
// that outlive the thread that uses them. All dereferences remain `unsafe` at
// the call site, so synchronization and aliasing guarantees stay explicit.
unsafe impl<T: Send> Send for SendPtr<T> {}
unsafe impl<T: Sync> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
  #[inline]
  pub(in crate::rminhash) const fn new(ptr: *mut T) -> Self {
    Self { ptr }
  }

  #[inline]
  pub(in crate::rminhash) const fn as_ptr(&self) -> *mut T {
    self.ptr
  }

  #[inline]
  pub(in crate::rminhash) const unsafe fn add(self, count: usize) -> *mut T {
    // SAFETY: caller ensures `count` stays within the allocated object.
    unsafe { self.ptr.add(count) }
  }
}
