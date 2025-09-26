pub mod table64;
pub mod lookup;
pub mod eight_value_lookup;

// Re-export the main types for convenience
pub use lookup::{SimdLookup, ScalarLookup, HashLookup, Lookup, U8x8};
pub use eight_value_lookup::EightValueLookup;
