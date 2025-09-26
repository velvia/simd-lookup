use simd_lookup::EightValueLookup;
use simd_aligned::arch::u32x8;

fn main() {
    println!("SIMD Eight-Value Position Lookup Demo");
    println!("====================================");

    // Create a lookup table with 8 values
    let lookup_table = [10, 20, 30, 40, 50, 60, 70, 80];
    let simd_lookup = EightValueLookup::new(&lookup_table);

    println!("Lookup table: {:?}", lookup_table);
    println!();

    // Test single value position lookup
    println!("Single value position tests:");
    let test_values = [10, 15, 20, 25, 30, 35, 90];
    for &val in &test_values {
        let position = simd_lookup.find_position(val);
        if position >= 0 {
            println!("  {} found at position {}", val, position);
        } else {
            println!("  {} not found (position -1)", val);
        }
    }
    println!();

    // Test batch position lookup using SIMD
    println!("Batch SIMD position test:");
    let batch_values = u32x8::from([10, 15, 20, 25, 30, 35, 40, 45]);
    let batch_results = simd_lookup.find_positions_batch(batch_values);

    let batch_array = batch_values.to_array();
    println!("  Input values: {:?}", batch_array);
    println!("  Positions:    {:?}", batch_results);
    println!();

    // Performance comparison example
    println!("Performance comparison (1M position lookups):");

    let test_data: Vec<u32> = (0..1_000_000).map(|i| (i * 7 + 13) % 100).collect();

    // SIMD approach
    let start = std::time::Instant::now();
    let mut simd_found_count = 0;
    for &val in &test_data {
        if simd_lookup.find_position(val) >= 0 {
            simd_found_count += 1;
        }
    }
    let simd_duration = start.elapsed();

    // Scalar approach
    let start = std::time::Instant::now();
    let mut scalar_found_count = 0;
    for &val in &test_data {
        if lookup_table.iter().position(|&x| x == val).is_some() {
            scalar_found_count += 1;
        }
    }
    let scalar_duration = start.elapsed();

    println!("  SIMD:   {:?} ({} found)", simd_duration, simd_found_count);
    println!("  Scalar: {:?} ({} found)", scalar_duration, scalar_found_count);

    if simd_duration < scalar_duration {
        let speedup = scalar_duration.as_nanos() as f64 / simd_duration.as_nanos() as f64;
        println!("  SIMD is {:.2}x faster!", speedup);
    } else {
        let slowdown = simd_duration.as_nanos() as f64 / scalar_duration.as_nanos() as f64;
        println!("  SIMD is {:.2}x slower (may need larger datasets or different CPU)", slowdown);
    }

    println!();
    println!("Use cases for SIMD position lookup:");
    println!("- Finding indices of values in small lookup tables");
    println!("- Fast enum/category mapping operations");
    println!("- Database-style position queries on small sets");
    println!("- Protocol ID to handler index mapping");
    println!("- Game entity type to behavior index mapping");
    println!("- Color palette index lookups");
}
