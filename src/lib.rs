use std::iter::{Flatten, Take};

fn dezig(value: i32) -> i32 {
    if (value & 1) != 0 {
        !(value >> 1)
    } else {
        value >> 1
    }
}

#[test]
fn test_dezig() {
    assert_eq!(dezig(0), 0);
    assert_eq!(dezig(1), -1);
    assert_eq!(dezig(2), 1);
    assert_eq!(dezig(3), -2);
    assert_eq!(dezig(4), 2);
}

pub struct TriangleIterator<'a> {
    data: &'a [u8],
    codes: &'a [u8],
    codeaux: &'a [u8],
    next: u32,
    last: i32,
    edge_fifo: Fifo<(u32, u32)>,
    vertex_fifo: Fifo<u32>,
}

impl<'a> TriangleIterator<'a> {
    pub fn new(bytes: &'a [u8], num_vertices: usize) -> Option<Self> {
        let bytes = bytes.get(1..)?;

        let num_triangles = num_vertices / 3;

        Some(Self {
            codes: bytes.get(..num_triangles)?,
            data: bytes.get(num_triangles..)?,
            codeaux: bytes.get(bytes.len().checked_sub(16)?..)?,
            next: 0,
            last: 0,
            edge_fifo: Default::default(),
            vertex_fifo: Default::default(),
        })
    }
}

impl<'a> TriangleIterator<'a> {
    fn next_index(&mut self) -> u32 {
        let value = self.next;
        self.next += 1;
        value
    }

    fn decode_index(&mut self) -> Option<u32> {
        let value = leb128::read::unsigned(&mut self.data).ok()?;

        let delta = dezig(value as i32);

        self.last = self.last.checked_add(delta)?;

        Some(self.last as u32)
    }
}

impl<'a> Iterator for TriangleIterator<'a> {
    type Item = [u32; 3];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (&code, codes) = self.codes.split_first()?;
        self.codes = codes;

        let (x, y) = split_byte(code);

        Some(if x < 0xf {
            let (a, b) = self.edge_fifo.values[x as usize];

            let c = if y == 0 {
                // Encodes a recently encountered edge and a next vertex
                let c = self.next_index();

                self.vertex_fifo.push(c);

                c
            } else if y < 0xd {
                // Encodes a recently encountered edge and a recently encountered vertex
                let c = self.vertex_fifo.values[y as usize];

                c
            } else if y == 0xd || y == 0xe {
                // Encodes a recently encountered edge and a vertex that's adjacent to last.
                self.last = if y == 0xd {
                    self.last - 1
                } else {
                    self.last + 1
                };

                let c = self.last as u32;
                self.vertex_fifo.push(c);
                c
            } else {
                // Encodes a recently encountered edge and a free-standing vertex encoded explicitly
                let c = self.decode_index()?;
                self.vertex_fifo.push(c);
                c
            };

            self.edge_fifo.push((c, b));
            self.edge_fifo.push((a, c));

            [a, b, c]
        } else {
            if y < 0xe {
                // Encodes three indices using codeaux table lookup and vertex FIFO.

                let (z, w) = split_byte(self.codeaux[y as usize]);

                let a = self.next_index();

                let b = if z == 0 {
                    self.next_index()
                } else {
                    self.vertex_fifo.values[z as usize - 1]
                };

                let c = if w == 0 {
                    self.next_index()
                } else {
                    self.vertex_fifo.values[w as usize - 1]
                };

                self.edge_fifo.push((b, a));
                self.edge_fifo.push((c, b));
                self.edge_fifo.push((a, c));

                self.vertex_fifo.push(a);

                if z == 0 {
                    self.vertex_fifo.push(b);
                }

                if w == 0 {
                    self.vertex_fifo.push(c);
                }

                [a, b, c]
            } else {
                // Encodes three indices explicitly.

                let (&byte, data) = self.data.split_first()?;
                self.data = data;

                let (z, w) = split_byte(byte);

                if (z, w) == (0, 0) {
                    self.next = 0;
                }

                let a = if y == 0xe {
                    self.next_index()
                } else {
                    self.decode_index()?
                };

                let b = if z == 0 {
                    self.next_index()
                } else if z == 0x0f {
                    self.decode_index()?
                } else {
                    self.vertex_fifo.values[z as usize - 1]
                };

                let c = if w == 0 {
                    self.next_index()
                } else if w == 0x0f {
                    self.decode_index()?
                } else {
                    self.vertex_fifo.values[w as usize - 1]
                };

                self.edge_fifo.push((b, a));
                self.edge_fifo.push((c, b));
                self.edge_fifo.push((a, c));

                self.vertex_fifo.push(a);
                if z == 0 || z == 0x0f {
                    self.vertex_fifo.push(b);
                }
                if w == 0 || w == 0x0f {
                    self.vertex_fifo.push(c);
                }

                [a, b, c]
            }
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Filter {
    Octahedral,
    Quaternion,
    Exponential,
}

pub struct AttributeIterator<'a, const BYTE_STRIDE: usize> {
    tail_data: [u8; BYTE_STRIDE],
    attr_block_max_element_count: usize,
    bytes: &'a [u8],
    remaining_elements: usize,
    filter: Option<Filter>,
}

impl<'a, const BYTE_STRIDE: usize> AttributeIterator<'a, BYTE_STRIDE> {
    pub fn new(
        bytes: &'a [u8],
        element_count: usize,
        filter: Option<Filter>,
    ) -> Option<Take<Flatten<Self>>> {
        let attr_block_max_element_count = ((8192 / BYTE_STRIDE) % !15).min(256);

        if attr_block_max_element_count != 256 {
            panic!(
                "Expected a block element count of 255: {}",
                attr_block_max_element_count
            );
        }

        Some(
            Self {
                tail_data: (&bytes[bytes.len().checked_sub(BYTE_STRIDE)?..])
                    .try_into()
                    .ok()?,
                filter,
                bytes: &bytes[1..],
                attr_block_max_element_count,
                remaining_elements: element_count,
            }
            .flatten()
            .take(element_count),
        )
    }
}

impl<'a, const BYTE_STRIDE: usize> Iterator for AttributeIterator<'a, BYTE_STRIDE> {
    type Item = [[u8; BYTE_STRIDE]; 256];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_elements == 0 {
            return None;
        }

        let attr_block_element_count = self
            .remaining_elements
            .min(self.attr_block_max_element_count);

        let group_count = div_ceil(attr_block_element_count, 16);
        let header_byte_count = div_ceil(group_count, 4);

        let mut output_elements = [[0; BYTE_STRIDE]; 256];

        for byte in 0..BYTE_STRIDE {
            let header_bytes = self.bytes.get(..header_byte_count)?;
            self.bytes = self.bytes.get(header_byte_count..)?;

            for group in 0..group_count {
                let header_byte = header_bytes[group / 4];

                let mode = (header_byte >> ((group % 4) * 2)) & 0x3;

                let mut deltas = [0; 16];

                match mode {
                    0 => {}
                    1 => {
                        let sentinel_deltas = &self.bytes.get(..4)?;
                        self.bytes = self.bytes.get(4..)?;

                        for i in 0..16 {
                            let shift = 6 - ((i & 0x3) * 2);
                            let delta = (sentinel_deltas[i / 4] >> shift) & 0x3;
                            if delta == 0x3 {
                                let (&delta, bytes) = self.bytes.split_first()?;
                                deltas[i] = delta;
                                self.bytes = bytes;
                            } else {
                                deltas[i] = delta;
                            }
                        }
                    }
                    2 => {
                        let sentinel_deltas = &self.bytes.get(..8)?;
                        self.bytes = self.bytes.get(8..)?;
                        for i in 0..16 {
                            let shift = if i % 2 == 1 { 0 } else { 4 };

                            let delta = (sentinel_deltas[i / 2] >> shift) & 0xf;

                            if delta == 0xf {
                                let (&delta, bytes) = self.bytes.split_first()?;
                                deltas[i] = delta;
                                self.bytes = bytes;
                            } else {
                                deltas[i] = delta;
                            }
                        }
                    }
                    _ => {
                        deltas.copy_from_slice(self.bytes.get(..16)?);
                        self.bytes = self.bytes.get(16..)?;
                    }
                }

                for i in 0..16 {
                    let dst_elem = group * 16 + i;

                    self.tail_data[byte] =
                        (self.tail_data[byte] as i32 + dezig(deltas[i] as i32)) as u8;

                    output_elements[dst_elem][byte] = self.tail_data[byte];
                }
            }
        }

        if let Some(filter) = self.filter {
            for element in &mut output_elements[..self.remaining_elements.min(256)] {
                filter_element(element, BYTE_STRIDE, filter);
            }
        }

        self.remaining_elements -= attr_block_element_count;
        Some(output_elements)
    }
}

// Seems to be significantly faster than f32::round.
// Essentially does a simd mask select without the simd.
#[inline]
fn approx_round(value: f32) -> f32 {
    value - 0.5 + ((value >= 0.0) as u8 as f32)
}

#[inline]
fn filter_element(input: &mut [u8], byte_stride: usize, filter: Filter) {
    match filter {
        Filter::Octahedral => {
            let (mut x, mut y, one, max_int) = if byte_stride == 4 {
                (
                    (input[0] as i8) as f32,
                    (input[1] as i8) as f32,
                    (input[2] as i8) as f32,
                    127.0,
                )
            } else {
                (
                    i16::from_le_bytes([input[0], input[1]]) as f32,
                    i16::from_le_bytes([input[2], input[3]]) as f32,
                    i16::from_le_bytes([input[4], input[5]]) as f32,
                    32767.0,
                )
            };

            x /= one;
            y /= one;
            let mut z = 1.0 - x.abs() - y.abs();

            let t = z.min(0.0);

            x -= t.copysign(x);
            y -= t.copysign(y);

            let normalize_and_multiply = max_int / (x * x + y * y + z * z).sqrt();

            x *= normalize_and_multiply;
            y *= normalize_and_multiply;
            z *= normalize_and_multiply;

            if byte_stride == 4 {
                input[0] = approx_round(x) as i8 as u8;
                input[1] = approx_round(y) as i8 as u8;
                input[2] = approx_round(z) as i8 as u8;
            } else {
                let x = approx_round(x) as i16 as u16;
                input[0..2].copy_from_slice(&x.to_le_bytes());

                let y = approx_round(y) as i16 as u16;
                input[2..4].copy_from_slice(&y.to_le_bytes());

                let z = approx_round(z) as i16 as u16;
                input[4..6].copy_from_slice(&z.to_le_bytes());
            }
        }
        Filter::Quaternion => {
            // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_meshopt_compression#filter-2-quaternion

            let input_0 = i16::from_le_bytes([input[0], input[1]]);
            let input_1 = i16::from_le_bytes([input[2], input[3]]);
            let input_2 = i16::from_le_bytes([input[4], input[5]]);
            let input_3 = i16::from_le_bytes([input[6], input[7]]);

            let one = (input_3 | 3) as f32;

            let normalize = std::f32::consts::FRAC_1_SQRT_2 / one;

            let x = input_0 as f32 * normalize;
            let y = input_1 as f32 * normalize;
            let z = input_2 as f32 * normalize;

            let w = (0.0_f32).max(1.0 - x * x - y * y - z * z).sqrt();

            let x = (approx_round(x * 32767.0) as i16).to_le_bytes();
            let y = (approx_round(y * 32767.0) as i16).to_le_bytes();
            let z = (approx_round(z * 32767.0) as i16).to_le_bytes();
            let w = (approx_round(w * 32767.0) as i16).to_le_bytes();

            let maxcomp = input_3 & 3;

            let range_for_index = |index| {
                let start = (((maxcomp + 1 + index) % 4) * 2) as usize;
                start..start + 2
            };

            input[range_for_index(0)].copy_from_slice(&x);
            input[range_for_index(1)].copy_from_slice(&y);
            input[range_for_index(2)].copy_from_slice(&z);
            input[range_for_index(3)].copy_from_slice(&w);
        }
        Filter::Exponential => {
            for chunk in input.chunks_mut(4) {
                let int = i32::from_le_bytes(chunk.try_into().unwrap());
                let exponent = int >> 24;
                // sign extension.
                let mantissa = (int << 8) >> 8;

                let float = 2.0_f32.powi(exponent) * mantissa as f32;

                chunk.copy_from_slice(&float.to_le_bytes());
            }
        }
    }
}

#[inline]
fn div_ceil(dividend: usize, divisor: usize) -> usize {
    (dividend + (divisor - 1)) / divisor
}

#[test]
fn test_div_ceil() {
    assert_eq!(div_ceil(5, 2), 3);
}

#[inline]
fn split_byte(byte: u8) -> (u8, u8) {
    (byte >> 4, byte & 0x0f)
}

#[derive(Default, Debug)]
struct Fifo<T> {
    values: [T; 16],
}

impl<T: Copy> Fifo<T> {
    fn push(&mut self, value: T) {
        self.values.copy_within(0..15, 1);
        self.values[0] = value;
    }
}

#[test]
fn decode_index_buffer() {
    let encoded = [
        0xe1, 0xf0, 0x10, 0xfe, 0x1f, 0x3d, 0x00, 0x0a, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9,
        0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
    ];

    let expected = [0, 1, 2, 2, 1, 3, 0, 1, 2, 2, 1, 5, 2, 1, 4];

    let result: Vec<_> = TriangleIterator::new(&encoded, 15)
        .unwrap()
        .flatten()
        .collect();

    assert_eq!(result, expected);
}

#[test]
fn decode_index_buffer_more() {
    let encoded = [
        225, 240, 16, 254, 255, 240, 12, 255, 2, 2, 2, 0, 118, 135, 86, 103, 120, 169, 134, 101,
        137, 104, 152, 1, 105, 0, 0,
    ];

    let expected = [0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9];

    let result: Vec<_> = TriangleIterator::new(&encoded, 12)
        .unwrap()
        .flatten()
        .collect();

    assert_eq!(result, expected);
}

#[test]
fn decode_index_buffer_3_edges() {
    let encoded = [
        0xe1, 0xf0, 0x20, 0x30, 0x40, 0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xa9, 0x86, 0x65, 0x89,
        0x68, 0x98, 0x01, 0x69, 0x00, 0x00,
    ];

    let expected = [0, 1, 2, 1, 0, 3, 2, 1, 4, 0, 2, 5];

    let result: Vec<_> = TriangleIterator::new(&encoded, 12)
        .unwrap()
        .flatten()
        .collect();

    assert_eq!(result, expected);
}

#[test]
fn decode_real_index_buffer() {
    let test_bytes = &std::fs::read("test_data/index_buffer.uncompressed.bin").unwrap();

    let comp_bytes = &std::fs::read("test_data/index_buffer.compressed.bin").unwrap();

    TriangleIterator::new(&comp_bytes, test_bytes.len() / 2)
        .unwrap()
        .zip(test_bytes.chunks(6))
        .for_each(|(tri, test)| {
            let test = [
                u16::from_le_bytes([test[0], test[1]]) as u32,
                u16::from_le_bytes([test[2], test[3]]) as u32,
                u16::from_le_bytes([test[4], test[5]]) as u32,
            ];

            let eq = tri == test
                || tri == [test[1], test[2], test[0]]
                || tri == [test[2], test[0], test[1]];

            if !eq {
                panic!("{:?} {:?}", &tri, &test);
            }
        })
}

#[test]
fn decode_vertex_buffer() {
    let encoded = [
        0xa0, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x58, 0x57, 0x58, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01,
        0x0c, 0x00, 0x00, 0x00, 0x58, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
        0x3f, 0x00, 0x00, 0x00, 0x17, 0x18, 0x17, 0x01, 0x26, 0x00, 0x00, 0x00, 0x01, 0x0c, 0x00,
        0x00, 0x00, 0x17, 0x01, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let expected = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 44, 1, 0, 0, 0, 0, 0, 0, 244, 1, 0, 0, 0, 0, 44, 1, 0,
        0, 0, 0, 0, 0, 244, 1, 44, 1, 44, 1, 0, 0, 0, 0, 244, 1, 244, 1,
    ];

    assert_eq!(
        AttributeIterator::<12>::new(&encoded, 4, None)
            .unwrap()
            .flatten()
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn decode_vertex_buffer_more() {
    let encoded = [
        160, 0, 1, 42, 170, 170, 170, 2, 4, 68, 68, 68, 68, 68, 68, 68, 3, 0, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    let expected = [
        0, 0, 0, 0, 0, 1, 2, 8, 0, 2, 4, 16, 0, 3, 6, 24, 0, 4, 8, 32, 0, 5, 10, 40, 0, 6, 12, 48,
        0, 7, 14, 56, 0, 8, 16, 64, 0, 9, 18, 72, 0, 10, 20, 80, 0, 11, 22, 88, 0, 12, 24, 96, 0,
        13, 26, 104, 0, 14, 28, 112, 0, 15, 30, 120,
    ];

    assert_eq!(
        AttributeIterator::<4>::new(&encoded, 16, None)
            .unwrap()
            .flatten()
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn decode_vertex_buffer_more_2() {
    let encoded = [
        0xa0, 0x02, 0x08, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x02, 0x0a, 0xaa, 0xaa, 0xaa,
        0xaa, 0xaa, 0xaa, 0xaa, 0x02, 0x0c, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0x02, 0x0e,
        0xee, 0xee, 0xee, 0xee, 0xee, 0xee, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    let expected = [
        0, 0, 0, 0, 4, 5, 6, 7, 8, 10, 12, 14, 12, 15, 18, 21, 16, 20, 24, 28, 20, 25, 30, 35, 24,
        30, 36, 42, 28, 35, 42, 49, 32, 40, 48, 56, 36, 45, 54, 63, 40, 50, 60, 70, 44, 55, 66, 77,
        48, 60, 72, 84, 52, 65, 78, 91, 56, 70, 84, 98, 60, 75, 90, 105,
    ];

    assert_eq!(
        AttributeIterator::<4>::new(&encoded, 16, None)
            .unwrap()
            .flatten()
            .collect::<Vec<_>>(),
        &expected
    );
}

#[test]
fn decode_filter_oct_8() {
    let encoded = [
        0xa0, 0x01, 0x07, 0x00, 0x00, 0x00, 0x1e, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x8b, 0x8c, 0xfd,
        0x00, 0x01, 0x26, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x7f, 0x00,
    ];

    let expected = [0, 1, 127, 0, 0, 159, 82, 1, 255, 1, 127, 0, 1, 130, 241, 1];

    assert_eq!(
        AttributeIterator::<4>::new(&encoded, 4, Some(Filter::Octahedral))
            .unwrap()
            .flatten()
            .collect::<Vec<_>>(),
        &expected
    );
}

#[test]
fn decode_filter_oct_12() {
    let encoded = [
        0xa0, 0x01, 0x0f, 0x00, 0x00, 0x00, 0x3d, 0x5a, 0x01, 0x0f, 0x00, 0x00, 0x00, 0x0e, 0x0d,
        0x01, 0x3f, 0x00, 0x00, 0x00, 0x9a, 0x99, 0x26, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x0e, 0x0d,
        0x0a, 0x00, 0x00, 0x01, 0x26, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0xff, 0x07, 0x00, 0x00,
    ];

    let expected = [
        0, 16, 32767, 0, 0, 32621, 3088, 1, 32764, 16, 471, 0, 307, 28541, 16093, 1,
    ];

    assert_eq!(
        AttributeIterator::<8>::new(&encoded, 4, Some(Filter::Octahedral))
            .unwrap()
            .flat_map(|bytes| [
                u16::from_le_bytes([bytes[0], bytes[1]]),
                u16::from_le_bytes([bytes[2], bytes[3]]),
                u16::from_le_bytes([bytes[4], bytes[5]]),
                u16::from_le_bytes([bytes[6], bytes[7]]),
            ])
            .collect::<Vec<_>>(),
        &expected
    );
}

#[test]
fn decode_filter_quat_12() {
    let encoded = [
        0xa0, 0x01, 0x0f, 0x00, 0x00, 0x00, 0x3d, 0x5a, 0x01, 0x0f, 0x00, 0x00, 0x00, 0x0e, 0x0d,
        0x01, 0x3f, 0x00, 0x00, 0x00, 0x9a, 0x99, 0x26, 0x01, 0x3f, 0x00, 0x00, 0x00, 0x0e, 0x0d,
        0x0a, 0x00, 0x00, 0x01, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xfc, 0x07,
    ];

    let expected = [
        32767, 0, 11, 0, 0, 25013, 0, 21166, 11, 0, 23504, 22830, 158, 14715, 0, 29277,
    ];

    assert_eq!(
        AttributeIterator::<8>::new(&encoded, 4, Some(Filter::Quaternion))
            .unwrap()
            .flat_map(|bytes| [
                u16::from_le_bytes([bytes[0], bytes[1]]),
                u16::from_le_bytes([bytes[2], bytes[3]]),
                u16::from_le_bytes([bytes[4], bytes[5]]),
                u16::from_le_bytes([bytes[6], bytes[7]]),
            ])
            .collect::<Vec<_>>(),
        &expected
    );
}

#[test]
fn decode_filter_exp() {
    let encoded = [
        0xa0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0xff, 0xf7, 0xff, 0xff, 0x02,
        0xff, 0xff, 0x7f, 0xfe,
    ];

    let expected = [0, 0x3fc00000, 0xc2100000, 0x49fffffe];

    assert_eq!(
        AttributeIterator::<16>::new(&encoded, 1, Some(Filter::Exponential))
            .unwrap()
            .flat_map(|bytes| [
                u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
                u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
                u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            ])
            .collect::<Vec<_>>(),
        &expected
    );
}

#[test]
fn decode_real_vertex_buffer() {
    let compressed_bytes = std::fs::read("test_data/vertex_buffer.compressed.bin").unwrap();

    assert_eq!(
        AttributeIterator::<4>::new(&compressed_bytes, 3678936, Some(Filter::Octahedral))
            .unwrap()
            .count(),
        3678936
    );
}

#[test]
fn decode_real_vertex_buffer_quat() {
    let compressed_bytes = std::fs::read("test_data/vertex_buffer_quat.compressed.bin").unwrap();

    assert_eq!(
        AttributeIterator::<8>::new(&compressed_bytes, 1788, Some(Filter::Quaternion))
            .unwrap()
            .count(),
        1788
    );
}
