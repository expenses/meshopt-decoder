use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_real_index_buffer(c: &mut Criterion) {
    let comp_bytes = std::fs::read("test_data/index_buffer.compressed.bin").unwrap();

    let num_indices = 54348;
    let num_trianges = num_indices / 3;

    c.bench_function("bench_real_index_buffer", |b| {
        b.iter(|| {
            let iter = meshopt_decoder::TriangleIterator::new(&comp_bytes, num_indices).unwrap();
            let mut got = 0;

            for tri in iter {
                black_box(tri);
                got += 1;
            }

            assert_eq!(got, num_trianges);
        })
    });
}

fn bench_real_vertex_buffer(c: &mut Criterion) {
    let compressed_bytes = std::fs::read("test_data/vertex_buffer.compressed.bin").unwrap();

    c.bench_function("bench_real_vertex_buffer", |b| {
        b.iter(|| {
            let mut count = 0;

            for x in meshopt_decoder::AttributeIterator::<4>::new(
                &compressed_bytes,
                3678936,
                Some(meshopt_decoder::Filter::Octahedral),
            )
            .unwrap()
            {
                black_box(x);
                count += 1;
            }

            assert_eq!(count, 3678936)
        })
    });
}

fn bench_real_vertex_buffer_quaternion(c: &mut Criterion) {
    let compressed_bytes = std::fs::read("test_data/vertex_buffer_quat.compressed.bin").unwrap();

    c.bench_function("bench_real_vertex_buffer_quaternion", |b| {
        b.iter(|| {
            let mut count = 0;

            for x in meshopt_decoder::AttributeIterator::<8>::new(
                &compressed_bytes,
                1788,
                Some(meshopt_decoder::Filter::Quaternion),
            )
            .unwrap()
            {
                black_box(x);
                count += 1;
            }

            assert_eq!(count, 1788)
        })
    });
}

criterion_group!(
    benches,
    bench_real_index_buffer,
    bench_real_vertex_buffer,
    bench_real_vertex_buffer_quaternion
);
criterion_main!(benches);
