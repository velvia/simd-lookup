use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simd_lookup::lookup::{HashLookup, Lookup, ScalarLookup, SimdLookup, U8x8};
use simd_aligned::arch::u32x8;

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

fn create_lookup_keys(max_key: u32, num_keys: usize) -> Vec<u32> {
    let mut keys = Vec::with_capacity(num_keys);
    for i in 0..num_keys {
        // Mix of valid and invalid keys for realistic testing
        let key = if i % 4 == 0 {
            // 25% invalid keys (beyond range)
            max_key + (i as u32)
        } else {
            // 75% keys within range (some valid, some invalid)
            (i as u32) % max_key
        };
        keys.push(key);
    }
    keys
}

fn bench_single_lookup(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let scalar_lookup = ScalarLookup::new(&entries);
    let hash_lookup = HashLookup::new(&entries);
    let simd_lookup = SimdLookup::new(&entries);

    let test_keys = create_lookup_keys(max_key, 1000);

    let mut group = c.benchmark_group("single_lookup");

    group.bench_function("scalar", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(scalar_lookup.lookup(black_box(key)));
            }
        })
    });

    group.bench_function("hash", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(hash_lookup.lookup(black_box(key)));
            }
        })
    });

    group.bench_function("simd_scalar", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(simd_lookup.lookup_scalar(black_box(key)));
            }
        })
    });

    group.finish();
}

fn bench_batch_lookup(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let scalar_lookup = ScalarLookup::new(&entries);
    let hash_lookup = HashLookup::new(&entries);

    let mut group = c.benchmark_group("batch_lookup");

    for batch_size in [64, 256, 1024, 4096] {
        let test_keys = create_lookup_keys(max_key, batch_size);
        let mut results = vec![0u8; batch_size];

        group.bench_with_input(
            BenchmarkId::new("scalar", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hash", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    hash_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );
    }

    group.finish();
}

fn bench_simd_u32x8_lookup(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let simd_lookup = SimdLookup::new(&entries);

    let test_keys = create_lookup_keys(max_key, 1024);

    // Convert to u32x8 chunks
    let mut u32x8_keys = Vec::new();
    for chunk in test_keys.chunks(8) {
        if chunk.len() == 8 {
            let array: [u32; 8] = chunk.try_into().unwrap();
            u32x8_keys.push(u32x8::from(array));
        }
    }

    let mut results = vec![U8x8::from([0; 8]); u32x8_keys.len()];

    let mut group = c.benchmark_group("simd_lookup");

    group.bench_function("u32x8_single", |b| {
        b.iter(|| {
            for &keys in &u32x8_keys {
                black_box(simd_lookup.lookup_u32x8(black_box(keys)));
            }
        })
    });

    group.bench_function("u32x8_batch", |b| {
        b.iter(|| {
            simd_lookup.lookup_batch_u32x8(black_box(&u32x8_keys), black_box(&mut results));
        })
    });

    group.finish();
}

fn bench_simd_vs_scalar_comparison(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let scalar_lookup = ScalarLookup::new(&entries);
    let simd_lookup = SimdLookup::new(&entries);

    let test_keys = create_lookup_keys(max_key, 1024);

    // Convert to u32x8 chunks for SIMD
    let mut u32x8_keys = Vec::new();
    for chunk in test_keys.chunks(8) {
        if chunk.len() == 8 {
            let array: [u32; 8] = chunk.try_into().unwrap();
            u32x8_keys.push(u32x8::from(array));
        }
    }

    let mut scalar_results = vec![0u8; test_keys.len()];
    let mut simd_results = vec![U8x8::from([0; 8]); u32x8_keys.len()];

    let mut group = c.benchmark_group("simd_vs_scalar");

    group.bench_function("scalar_batch", |b| {
        b.iter(|| {
            scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut scalar_results));
        })
    });

    group.bench_function("simd_batch", |b| {
        b.iter(|| {
            simd_lookup.lookup_batch_u32x8(black_box(&u32x8_keys), black_box(&mut simd_results));
        })
    });

    group.finish();
}

fn bench_density_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_comparison");

    for density in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let entries = create_sparse_entries(1_000_000, density);
        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

        let scalar_lookup = ScalarLookup::new(&entries);
        let hash_lookup = HashLookup::new(&entries);

        let test_keys = create_lookup_keys(max_key, 1000);
        let mut results = vec![0u8; 1000];

        group.bench_with_input(
            BenchmarkId::new("scalar", density),
            &density,
            |b, _| {
                b.iter(|| {
                    scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hash", density),
            &density,
            |b, _| {
                b.iter(|| {
                    hash_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_usage_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Test different table sizes to see cache effects
    for table_size in [10_000, 100_000, 1_000_000, 10_000_000] {
        let entries = create_sparse_entries(table_size, 1.0); // 1% density
        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

        let scalar_lookup = ScalarLookup::new(&entries);
        let hash_lookup = HashLookup::new(&entries);

        let test_keys = create_lookup_keys(max_key, 1000);
        let mut results = vec![0u8; 1000];

        group.bench_with_input(
            BenchmarkId::new("scalar", table_size),
            &table_size,
            |b, _| {
                b.iter(|| {
                    scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hash", table_size),
            &table_size,
            |b, _| {
                b.iter(|| {
                    hash_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_lookup,
    bench_batch_lookup,
    bench_simd_u32x8_lookup,
    bench_simd_vs_scalar_comparison,
    bench_density_comparison,
    bench_memory_usage_patterns
);
criterion_main!(benches);