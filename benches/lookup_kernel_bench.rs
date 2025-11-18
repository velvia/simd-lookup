use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::prelude::*;
use simd_aligned::arch::u8x16;
use simd_lookup::lookup_kernel::{SimdSingleVocabU32U8Lookup, SimdSingleVocabU32U8LookupGather, ScalarSingleVocabU32U8Lookup, SimdDualVocabU32U8Lookup};

/// Create a sparse lookup table with the specified size and density
/// Returns a Vec<u8> where density_percent of entries are nonzero
fn create_sparse_lookup_table(size: usize, density_percent: f32) -> Vec<u8> {
    let num_nonzero = ((size as f32) * (density_percent / 100.0)) as usize;
    let mut table = vec![0u8; size];

    // Create sparse entries distributed across the range
    let step = size / num_nonzero.max(1);
    for i in 0..num_nonzero {
        let idx = i * step;
        if idx < size {
            // Use values 1-255 to avoid 0 (which is the default)
            table[idx] = ((idx % 255) + 1) as u8;
        }
    }

    table
}

/// Create test values (u32 indices) for lookup
fn create_test_values(num_values: usize, max_index: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible benchmarks
    let mut values = Vec::with_capacity(num_values);

    for _ in 0..num_values {
        let idx = rng.gen_range(0..max_index);
        values.push(idx as u32);
    }

    values.shuffle(&mut rng);
    values
}

/// Benchmark SimdSingleVocabU32U8Lookup with chunks of 500
fn bench_single_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating lookup table: {} entries, {}% density", table_size, density);
    let lookup_table = create_sparse_lookup_table(table_size, density);
    let lookup = SimdSingleVocabU32U8Lookup::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("single_vocab_lookup");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500", |b| {
        // Pre-allocate to avoid repeated reserve() calls (2000 calls for 1M elements in chunks of 500)
        let mut result_vec = Vec::<u8x16>::with_capacity(num_values.div_ceil(16));
        b.iter(|| {
            // Reset the vec for each iteration
            result_vec.clear();
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                lookup.lookup_extend_u8x16_vec(black_box(chunk), &mut result_vec);
            }
            black_box(&result_vec);
        })
    });

    group.finish();
}

/// Benchmark SimdSingleVocabU32U8LookupGather with chunks of 500 (SIMD gather version)
fn bench_single_vocab_lookup_gather(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating lookup table (gather): {} entries, {}% density", table_size, density);
    let lookup_table = create_sparse_lookup_table(table_size, density);
    let lookup = SimdSingleVocabU32U8LookupGather::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("single_vocab_lookup_gather");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500", |b| {
        // Pre-allocate to avoid repeated reserve() calls (2000 calls for 1M elements in chunks of 500)
        let mut result_vec = Vec::<u8x16>::with_capacity(num_values.div_ceil(16));
        b.iter(|| {
            // Reset the vec for each iteration
            result_vec.clear();
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                lookup.lookup_extend_u8x16_vec(black_box(chunk), &mut result_vec);
            }
            black_box(&result_vec);
        })
    });

    group.finish();
}

/// Benchmark ScalarSingleVocabU32U8Lookup with chunks of 500 (scalar-only, no u8x16 SIMD)
fn bench_single_vocab_lookup_scalar(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating lookup table (scalar): {} entries, {}% density", table_size, density);
    let lookup_table = create_sparse_lookup_table(table_size, density);
    let lookup = ScalarSingleVocabU32U8Lookup::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("single_vocab_lookup_scalar");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500", |b| {
        b.iter(|| {
            // Process in chunks of 500, calling lookup_into_vec for each chunk
            let mut all_results = Vec::new();
            for chunk in test_values.chunks_exact(500) {
                let result = lookup.lookup_into_vec(black_box(chunk));
                all_results.extend(result);
            }
            black_box(all_results);
        })
    });

    group.finish();
}

/// Benchmark SimdDualVocabU32U8Lookup with chunks of 500
/// Takes bitwise AND of the two lookup results
fn bench_dual_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating dual lookup tables: {} entries, {}% density", table_size, density);
    let lookup_table1 = create_sparse_lookup_table(table_size, density);
    let lookup_table2 = create_sparse_lookup_table(table_size, density);
    let lookup = SimdDualVocabU32U8Lookup::new(&lookup_table1, &lookup_table2);

    // Create 1 million test values (two sets, divisible by 500)
    let num_values = 1_000_000;
    let test_values1 = create_test_values(num_values, table_size);
    let test_values2 = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("dual_vocab_lookup");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500_bitwise_and", |b| {
        b.iter(|| {
            let mut all_results = Vec::new();
            // Process in chunks of 500
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                // Use bitwise AND as the combiner function
                let result = lookup.lookup_into_vec(
                    black_box(chunk1),
                    black_box(chunk2),
                    &mut |v1, v2| v1 & v2
                );
                all_results.extend(result);
            }
            black_box(all_results);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_vocab_lookup,
    bench_single_vocab_lookup_gather,
    bench_single_vocab_lookup_scalar,
    bench_dual_vocab_lookup
);
criterion_main!(benches);

