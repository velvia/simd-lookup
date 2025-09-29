use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use simd_lookup::lookup::{HashLookup, Lookup, ScalarLookup};
use simd_lookup::entropy_map_lookup::{EntropyMapLookup, EntropyMapBitpackedLookup};
use std::time::Instant;

/// Create sparse entries for benchmarking
fn create_sparse_entries(size: usize, density_percent: f32) -> Vec<(u32, u8)> {
    let num_entries = ((size as f32) * (density_percent / 100.0)) as usize;
    let mut entries = Vec::with_capacity(num_entries);

    // Create sparse entries distributed across the range
    let step = size / num_entries.max(1);
    for i in 0..num_entries {
        let key = (i * step) as u32;
        let value = ((key % 255) + 1) as u8; // Values 1-255, avoiding 0 which is default
        entries.push((key, value));
    }

    entries
}

/// Create lookup keys for testing
fn create_lookup_keys(max_key: u32, num_keys: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible benchmarks
    let mut keys = Vec::with_capacity(num_keys);

    for _ in 0..num_keys {
        let key = rng.gen_range(0..=max_key);
        keys.push(key);
    }

    keys.shuffle(&mut rng);
    keys
}

/// Benchmark construction time for different map sizes
fn bench_construction_time(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_time");

    // Test sizes: 500, 1K, 20K, 100K as requested
    let sizes = [500, 1_000, 20_000, 100_000];
    let density = 1.0; // 1% density

    for &size in &sizes {
        let entries = create_sparse_entries(size, density);

        group.throughput(Throughput::Elements(entries.len() as u64));

        // Benchmark FxHashMap construction (baseline)
        group.bench_with_input(
            BenchmarkId::new("rustc_hash", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let _lookup = HashLookup::new(black_box(&entries));
                })
            },
        );

        // Benchmark ScalarLookup construction (for comparison)
        group.bench_with_input(
            BenchmarkId::new("scalar", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let _lookup = ScalarLookup::new(black_box(&entries));
                })
            },
        );

        // Benchmark EntropyMapLookup construction
        group.bench_with_input(
            BenchmarkId::new("entropy_map_dict", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let (_lookup, _time) = EntropyMapLookup::new(black_box(&entries));
                })
            },
        );

        // Benchmark EntropyMapBitpackedLookup construction
        group.bench_with_input(
            BenchmarkId::new("entropy_map_bitpacked", size),
            &size,
            |b, _| {
                b.iter(|| {
                    let (_lookup, _time) = EntropyMapBitpackedLookup::new(black_box(&entries));
                })
            },
        );
    }

    group.finish();
}

/// Benchmark lookup performance at large table size (20M) with low density
fn bench_large_table_lookup(c: &mut Criterion) {
    let table_size = 20_000_000; // 20 million
    let density = 1.0; // 1% density = 200K entries

    println!("Creating large table: {} entries, {}% density", table_size, density);
    let entries = create_sparse_entries(table_size, density);
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    println!("Building lookup structures...");

    // Build all lookup structures
    let hash_lookup = HashLookup::new(&entries);
    let scalar_lookup = ScalarLookup::new(&entries);
    let (entropy_dict_lookup, dict_construction_time) = EntropyMapLookup::new(&entries);
    let (entropy_bitpacked_lookup, bitpacked_construction_time) = EntropyMapBitpackedLookup::new(&entries);

    println!("Construction times:");
    println!("  EntropyMap Dict: {}ms", dict_construction_time / 1_000_000);
    println!("  EntropyMap Bitpacked: {}ms", bitpacked_construction_time / 1_000_000);

    // Create test keys
    let test_keys = create_lookup_keys(max_key, 500_000);
    let mut results = vec![0u8; test_keys.len()];

    let mut group = c.benchmark_group("large_table_lookup");
    group.throughput(Throughput::Elements(test_keys.len() as u64));

    group.bench_function("rustc_hash", |b| {
        b.iter(|| {
            hash_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
        })
    });

    group.bench_function("scalar", |b| {
        b.iter(|| {
            scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
        })
    });

    group.bench_function("entropy_map_dict", |b| {
        b.iter(|| {
            entropy_dict_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
        })
    });

    group.bench_function("entropy_map_bitpacked", |b| {
        b.iter(|| {
            entropy_bitpacked_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
        })
    });

    group.finish();
}

/// Benchmark single lookup performance at large table size
fn bench_large_table_single_lookup(c: &mut Criterion) {
    let table_size = 20_000_000; // 20 million
    let density = 1.0; // 1% density

    let entries = create_sparse_entries(table_size, density);
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    // Build all lookup structures
    let hash_lookup = HashLookup::new(&entries);
    let scalar_lookup = ScalarLookup::new(&entries);
    let (entropy_dict_lookup, _) = EntropyMapLookup::new(&entries);
    let (entropy_bitpacked_lookup, _) = EntropyMapBitpackedLookup::new(&entries);

    // Create test keys
    let test_keys = create_lookup_keys(max_key, 500_000);

    let mut group = c.benchmark_group("large_table_single_lookup");
    group.throughput(Throughput::Elements(test_keys.len() as u64));

    group.bench_function("rustc_hash", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(hash_lookup.lookup(black_box(key)));
            }
        })
    });

    group.bench_function("scalar", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(scalar_lookup.lookup(black_box(key)));
            }
        })
    });

    group.bench_function("entropy_map_dict", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(entropy_dict_lookup.lookup(black_box(key)));
            }
        })
    });

    group.bench_function("entropy_map_bitpacked", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(entropy_bitpacked_lookup.lookup(black_box(key)));
            }
        })
    });

    group.finish();
}

/// Benchmark lookup latency scaling with map size
fn bench_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("size_scaling");

    // Test different sizes to see how lookup performance scales
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let density = 1.0; // 1% density

    for &size in &sizes {
        let entries = create_sparse_entries(size, density);
        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

        // Build lookup structures
        let hash_lookup = HashLookup::new(&entries);
        let (entropy_dict_lookup, _) = EntropyMapLookup::new(&entries);
        let (entropy_bitpacked_lookup, _) = EntropyMapBitpackedLookup::new(&entries);

        // Create test keys
        let test_keys = create_lookup_keys(max_key, 500_000);

        group.throughput(Throughput::Elements(test_keys.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("rustc_hash", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for &key in &test_keys {
                        black_box(hash_lookup.lookup(black_box(key)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("entropy_map_dict", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for &key in &test_keys {
                        black_box(entropy_dict_lookup.lookup(black_box(key)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("entropy_map_bitpacked", size),
            &size,
            |b, _| {
                b.iter(|| {
                    for &key in &test_keys {
                        black_box(entropy_bitpacked_lookup.lookup(black_box(key)));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory usage and construction time analysis
fn bench_memory_and_construction_analysis(c: &mut Criterion) {
    let sizes = [500, 1_000, 5_000, 10_000, 20_000, 50_000, 100_000];
    let density = 1.0; // 1% density

    println!("\n=== Memory Usage and Construction Time Analysis ===");
    println!("{:<10} {:<12} {:<15} {:<15} {:<15} {:<15}",
             "Size", "Entries", "Hash(ms)", "Dict(ms)", "Bitpack(ms)", "Memory Est.");

    for &size in &sizes {
        let entries = create_sparse_entries(size, density);
        let num_entries = entries.len();

        // Measure construction times
        let start = Instant::now();
        let _hash_lookup = HashLookup::new(&entries);
        let hash_time = start.elapsed().as_millis();

        let start = Instant::now();
        let (entropy_dict_lookup, _) = EntropyMapLookup::new(&entries);
        let dict_time = start.elapsed().as_millis();

        let start = Instant::now();
        let (entropy_bitpacked_lookup, _) = EntropyMapBitpackedLookup::new(&entries);
        let bitpacked_time = start.elapsed().as_millis();

        // Estimate memory usage (rough approximation)
        let hash_memory_kb = (num_entries * 13) / 1024; // ~13 bytes per entry for FxHashMap
        let dict_memory_kb = entropy_dict_lookup.memory_usage() / 1024;
        let bitpacked_memory_kb = entropy_bitpacked_lookup.memory_usage() / 1024;

        println!("{:<10} {:<12} {:<15} {:<15} {:<15} {:<15}",
                 size, num_entries, hash_time, dict_time, bitpacked_time,
                 format!("H:{}KB D:{}KB B:{}KB", hash_memory_kb, dict_memory_kb, bitpacked_memory_kb));
    }

    // Run a simple benchmark to keep criterion happy
    let mut group = c.benchmark_group("memory_analysis");
    let entries = create_sparse_entries(1000, density);

    group.bench_function("dummy", |b| {
        b.iter(|| {
            let (_lookup, _time) = EntropyMapLookup::new(black_box(&entries));
        })
    });

    group.finish();
}

/// Benchmark different density levels
fn bench_density_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_comparison");

    let table_size = 1_000_000; // 1M table size
    let densities = [0.1, 0.5, 1.0, 2.0, 5.0]; // Different density percentages

    for &density in &densities {
        let entries = create_sparse_entries(table_size, density);
        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

        // Build lookup structures
        let hash_lookup = HashLookup::new(&entries);
        let (entropy_dict_lookup, _) = EntropyMapLookup::new(&entries);

        // Create test keys
        let test_keys = create_lookup_keys(max_key, 500_000);

        group.throughput(Throughput::Elements(test_keys.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("rustc_hash", density),
            &density,
            |b, _| {
                b.iter(|| {
                    for &key in &test_keys {
                        black_box(hash_lookup.lookup(black_box(key)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("entropy_map_dict", density),
            &density,
            |b, _| {
                b.iter(|| {
                    for &key in &test_keys {
                        black_box(entropy_dict_lookup.lookup(black_box(key)));
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_construction_time,
    bench_large_table_lookup,
    bench_large_table_single_lookup,
    bench_size_scaling,
    bench_memory_and_construction_analysis,
    bench_density_comparison
);
criterion_main!(benches);
