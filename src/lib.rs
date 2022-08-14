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
