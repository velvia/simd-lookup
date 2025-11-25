//! [BulkVecExtender] is a simple utility trait that allows you to bulk extend a Vec<T>
//! and return a `&mut [T]` slice that you can write to - much faster than individual `push()` calls,
//! which has to check for both bounds and capacity.
//!
use std::ops::{Deref, DerefMut};
use wide::u8x16;

/// [BulkVecExtender] is a simple utility trait that allows you to bulk extend a Vec<T>
/// and return a `&mut [T]` slice that you can write to - much faster than individual `push()` calls,
/// which has to check for both bounds and capacity.
///
/// # Example
///
/// Instead of using `push()` in a hot loop (which is expensive due to bounds checks):
///
/// ```rust,ignore
/// // Slow: each push() does bounds checking
/// let mut vec = Vec::new();
/// for value in some_data {
///     if condition(value) {
///         vec.push(value);  // Bounds check on every call!
///     }
/// }
/// ```
///
/// Use `bulk_extend_guard()` to get a RAII guard and write directly:
///
/// ```rust
/// # use simd_lookup::bulk_vec_extender::BulkVecExtender;
/// # let some_data = [1u8, 2, 3, 4, 5, 6, 7, 8];
/// # let condition = |v: u8| v % 2 == 0;
/// let mut vec = Vec::new();
/// let max_elements = 100;
///
/// {
///     // Get a guard to write to (no bounds checks during writes!)
///     let mut guard = vec.bulk_extend_guard(max_elements);
///
///     // Write directly to the guard - much faster than push()
///     let mut written = 0;
///     for value in some_data.iter().take(max_elements) {
///         if condition(*value) {
///             guard[written] = *value;
///             written += 1;
///         }
///     }
///
///     // Set actual number of elements written (only needed for partial writes)
///     guard.set_written(written);
///     // guard drops here, vec is automatically truncated to correct length
/// }
/// ```
///
/// # Performance
///
/// This trait is designed to eliminate the overhead of `Vec::push()` in hot loops:
/// - **No bounds checking** during writes (you get a pre-allocated slice)
/// - **Bulk allocation** happens once, not per-element
/// - **Better for SIMD** - you can write entire SIMD vectors at once
/// - **Cache-friendly** - sequential writes to a pre-allocated buffer
///
/// Benchmarks show `Vec::push()` can cost 35% of total performance in hot loops,
/// even with pre-allocation. This trait eliminates that overhead.
pub trait BulkVecExtender<T> {
    /// Returns a RAII guard that extends the Vec and automatically finalizes on drop.
    ///
    /// This is the preferred method for most use cases as it avoids borrow checker issues
    /// and automatically handles finalization.
    ///
    /// By default, assumes all elements will be written. If you write fewer elements,
    /// call `guard.set_written(count)` before the guard drops.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use simd_lookup::bulk_vec_extender::BulkVecExtender;
    /// let mut vec: Vec<u8> = Vec::new();
    /// {
    ///     let mut guard = vec.bulk_extend_guard(10);
    ///     for i in 0..10 {
    ///         guard[i] = i as u8;
    ///     }
    ///     // guard drops here, vec length is automatically set to 10
    /// }
    /// assert_eq!(vec.len(), 10);
    /// ```
    ///
    /// # Example with partial writes
    ///
    /// ```rust
    /// # use simd_lookup::bulk_vec_extender::BulkVecExtender;
    /// let mut vec: Vec<u8> = Vec::new();
    /// {
    ///     let mut guard = vec.bulk_extend_guard(100);
    ///     guard[0] = 42;
    ///     guard[1] = 43;
    ///     guard.set_written(2);  // only wrote 2 elements
    /// }
    /// assert_eq!(vec.len(), 2);
    /// ```
    fn bulk_extend_guard(&mut self, elements_to_write: usize) -> BulkExtendGuard<'_, T>;
}

/// RAII guard for bulk Vec extension. Automatically finalizes on drop.
///
/// When dropped, truncates the Vec to `original_len + written` elements.
/// By default, `written` equals the requested extension size, so if you
/// write all elements, you don't need to do anything special.
///
/// Use `set_written()` if you wrote fewer elements than the slice length.
pub struct BulkExtendGuard<'a, T> {
    vec: &'a mut Vec<T>,
    original_len: usize,
    extended_by: usize,
    written: usize,
}

impl<'a, T> BulkExtendGuard<'a, T> {
    /// Creates a new guard, extending the vec by `elements_to_write` elements.
    #[inline(always)]
    fn new(vec: &'a mut Vec<T>, elements_to_write: usize) -> Self {
        let original_len = vec.len();
        let new_len = original_len + elements_to_write;
        vec.reserve(elements_to_write);
        // Safety: we will finalize to the correct length on drop
        unsafe { vec.set_len(new_len); }
        Self {
            vec,
            original_len,
            extended_by: elements_to_write,
            written: elements_to_write,  // default: assume all elements will be written
        }
    }

    /// Set the actual number of elements written.
    /// Call this if you wrote fewer elements than the slice length.
    /// The count is capped to the extended size.
    #[inline(always)]
    pub fn set_written(&mut self, count: usize) {
        self.written = count.min(self.extended_by);
    }

    /// Returns the extended region as a mutable slice.
    /// This is a convenience method equivalent to `&mut *guard`.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.vec[self.original_len..]
    }
}

impl<T> Deref for BulkExtendGuard<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.vec[self.original_len..]
    }
}

impl<T> DerefMut for BulkExtendGuard<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec[self.original_len..]
    }
}

impl<T> Drop for BulkExtendGuard<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        self.vec.truncate(self.original_len + self.written);
    }
}

impl<T> BulkVecExtender<T> for Vec<T> {
    #[inline(always)]
    fn bulk_extend_guard(&mut self, elements_to_write: usize) -> BulkExtendGuard<'_, T> {
        BulkExtendGuard::new(self, elements_to_write)
    }
}

/// Utility trait to help write u8 SIMD vectors into a mutable slice
pub trait SliceU8SIMDExtender {
    /// Writes slice_len bytes of the u8x16 into a u8 mut slice at index.
    /// Panics if the slice does not have enough room (must have at least index+slice_len bytes).
    fn write_u8x16(&mut self, index: usize, value: u8x16, slice_len: usize);
}

impl SliceU8SIMDExtender for &mut [u8] {
    #[inline(always)]
    fn write_u8x16(&mut self, index: usize, value: u8x16, slice_len: usize) {
        self[index..index+slice_len].copy_from_slice(&value.as_array()[..slice_len]);
    }
}