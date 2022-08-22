use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_group(c: &mut Criterion) {
    c.bench_function("bench_real_index_buffer", |b| {
        let comp_bytes = std::fs::read("test_data/triangles_comp.bin").unwrap();

        let num_indices = 4995564;
        let num_trianges = num_indices / 3;

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

    c.bench_function("bench_real_vertex_buffer_positions", |b| {
        let compressed_bytes = std::fs::read("test_data/positions_comp.bin").unwrap();

        b.iter(|| {
            let mut count = 0;

            for x in meshopt_decoder::AttributeIterator::<8>::new(&compressed_bytes, 3678936, None)
                .unwrap()
            {
                black_box(x);
                count += 1;
            }

            assert_eq!(count, 3678936)
        })
    });

    c.bench_function("bench_real_vertex_buffer_normals", |b| {
        let compressed_bytes = std::fs::read("test_data/normals_comp.bin").unwrap();

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

    c.bench_function("bench_real_vertex_buffer_normals_vec", |b| {
        let compressed_bytes = std::fs::read("test_data/normals_comp.bin").unwrap();

        b.iter(|| {
            let vec = meshopt_decoder::decompress_attributes_to_vec(&compressed_bytes, 3678936, Some(meshopt_decoder::Filter::Octahedral), 4).unwrap();

            let vec = black_box(vec);

            assert_eq!(vec.len(), 3678936 * 4)
        })
    });

    c.bench_function("bench_real_vertex_buffer_uvs", |b| {
        let compressed_bytes = std::fs::read("test_data/uvs_comp.bin").unwrap();

        b.iter(|| {
            let mut count = 0;

            for x in meshopt_decoder::AttributeIterator::<4>::new(&compressed_bytes, 3678936, None)
                .unwrap()
            {
                black_box(x);
                count += 1;
            }

            assert_eq!(count, 3678936)
        })
    });

    c.bench_function("bench_real_vertex_buffer_octahedral", |b| {
        let compressed_bytes = std::fs::read("test_data/vertex_buffer.compressed.bin").unwrap();

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

    c.bench_function("bench_real_vertex_buffer_quaternion", |b| {
        let compressed_bytes =
            std::fs::read("test_data/vertex_buffer_quat.compressed.bin").unwrap();

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

criterion_group!(benches, bench_group,);
criterion_main!(benches);
