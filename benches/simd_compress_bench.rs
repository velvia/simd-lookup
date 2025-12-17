use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simd_lookup::simd_compress::{compress_store_u32x8, compress_store_u32x16, compress_store_u8x16};
use wide::{u32x8, u32x16, u8x16};
use rand::prelude::*;

// =============================================================================
// Test Data Generators
// =============================================================================

/// Generate random masks with approximately the given density (0.0 to 1.0)
fn generate_mask_u8(density: f64) -> u8 {
    let mut rng = thread_rng();
    let mut mask = 0u8;
    for i in 0..8 {
        if rng.gen_bool(density) {
            mask |= 1 << i;
        }
    }
    mask
}

/// Generate random masks with approximately the given density (0.0 to 1.0)
fn generate_mask_u16(density: f64) -> u16 {
    let mut rng = thread_rng();
    let mut mask = 0u16;
    for i in 0..16 {
        if rng.gen_bool(density) {
            mask |= 1 << i;
        }
    }
    mask
}

/// Generate test data vectors
fn generate_u32x8_data() -> u32x8 {
    u32x8::from([10, 20, 30, 40, 50, 60, 70, 80])
}

fn generate_u32x16_data() -> u32x16 {
    u32x16::from([
        10, 20, 30, 40, 50, 60, 70, 80,
        90, 100, 110, 120, 130, 140, 150, 160
    ])
}

fn generate_u8x16_data() -> u8x16 {
    u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

// =============================================================================
// Benchmarks: compress_store (store to slice)
// =============================================================================

fn bench_compress_store_u32x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_store_u32x8");

    let data = generate_u32x8_data();
    let densities = [0.1, 0.25, 0.5, 0.75, 0.9];

    for density in densities.iter() {
        let mask = generate_mask_u8(*density);
        let count = mask.count_ones() as usize;
        let mut output = vec![0u32; 8];

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("density", format!("{:.0}%", density * 100.0)),
            &mask,
            |b, &mask| {
                b.iter(|| {
                    compress_store_u32x8(black_box(data), black_box(mask), black_box(&mut output));
                });
            },
        );
    }

    group.finish();
}

fn bench_compress_store_u8x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_store_u8x16");

    let data = generate_u8x16_data();
    let densities = [0.1, 0.25, 0.5, 0.75, 0.9];

    for density in densities.iter() {
        let mask = generate_mask_u16(*density);
        let count = mask.count_ones() as usize;
        let mut output = vec![0u8; 16];

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("density", format!("{:.0}%", density * 100.0)),
            &mask,
            |b, &mask| {
                b.iter(|| {
                    compress_store_u8x16(black_box(data), black_box(mask), black_box(&mut output));
                });
            },
        );
    }

    group.finish();
}

fn bench_compress_store_u32x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_store_u32x16");

    let data = generate_u32x16_data();
    let densities = [0.1, 0.25, 0.5, 0.75, 0.9];

    for density in densities.iter() {
        let mask = generate_mask_u16(*density);
        let count = mask.count_ones() as usize;
        let mut output = vec![0u32; 16];

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("density", format!("{:.0}%", density * 100.0)),
            &mask,
            |b, &mask| {
                b.iter(|| {
                    compress_store_u32x16(black_box(data), black_box(mask), black_box(&mut output));
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark: Batch operations (simulating real-world usage)
// =============================================================================

fn bench_compress_batch_u32x8(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_batch_u32x8");

    let data = generate_u32x8_data();
    let batch_size = 1000;
    let mut masks = Vec::with_capacity(batch_size);
    let mut outputs = vec![vec![0u32; 8]; batch_size];

    // Generate random masks with varying densities
    for _ in 0..batch_size {
        masks.push(generate_mask_u8(0.5)); // 50% average density
    }

    group.throughput(Throughput::Elements(batch_size as u64));
    group.bench_function("batch_1000", |b| {
        b.iter(|| {
            for i in 0..batch_size {
                let mask = masks[i];
                compress_store_u32x8(
                    black_box(data),
                    black_box(mask),
                    black_box(&mut outputs[i]),
                );
            }
        });
    });

    group.finish();
}

fn bench_compress_batch_u8x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_batch_u8x16");

    let data = generate_u8x16_data();
    let batch_size = 1000;
    let mut masks = Vec::with_capacity(batch_size);
    let mut outputs = vec![vec![0u8; 16]; batch_size];

    // Generate random masks with varying densities
    for _ in 0..batch_size {
        masks.push(generate_mask_u16(0.5)); // 50% average density
    }

    group.throughput(Throughput::Elements(batch_size as u64));
    group.bench_function("batch_1000", |b| {
        b.iter(|| {
            for i in 0..batch_size {
                let mask = masks[i];
                compress_store_u8x16(
                    black_box(data),
                    black_box(mask),
                    black_box(&mut outputs[i]),
                );
            }
        });
    });

    group.finish();
}

// =============================================================================
// Main
// =============================================================================

criterion_group!(
    benches,
    bench_compress_store_u32x8,
    bench_compress_store_u8x16,
    bench_compress_store_u32x16,
    bench_compress_batch_u32x8,
    bench_compress_batch_u8x16,
);
criterion_main!(benches);

