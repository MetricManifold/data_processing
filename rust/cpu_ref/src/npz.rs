//! Minimal NPY/NPZ writer (NumPy v1.0 format).
//!
//! Produces files identical (header-padded to 64 bytes) to those written by
//! `numpy.savez`, so `np.load("...npz")["key"]` works directly.

use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::Result;
use zip::write::SimpleFileOptions;
use zip::ZipWriter;

fn shape_str(shape: &[usize]) -> String {
    if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|n| n.to_string()).collect();
        format!("({})", parts.join(", "))
    }
}

/// Write an NPY-formatted byte stream to `out`.
fn write_npy_to<W: Write>(
    out: &mut W,
    descr: &str,
    shape: &[usize],
    data: &[u8],
) -> Result<()> {
    out.write_all(b"\x93NUMPY")?;
    out.write_all(&[0x01, 0x00])?; // version 1.0
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr,
        shape_str(shape)
    );
    // Total prefix = 10 bytes (magic 6 + ver 2 + header_len 2). NumPy requires
    // (10 + len(header_dict) + pad + 1 newline) be a multiple of 64.
    let raw = header_dict.into_bytes();
    let unpadded = 10 + raw.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    let header_len = raw.len() + pad + 1;
    out.write_all(&(header_len as u16).to_le_bytes())?;
    out.write_all(&raw)?;
    out.write_all(&vec![b' '; pad])?;
    out.write_all(b"\n")?;
    out.write_all(data)?;
    Ok(())
}

/// In-memory NPY blob for a slice of plain f64 / f32 / i32.
fn npy_bytes<T: Copy>(descr: &str, shape: &[usize], data: &[T]) -> Result<Vec<u8>> {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };
    let mut buf = Vec::with_capacity(128 + bytes.len());
    write_npy_to(&mut buf, descr, shape, bytes)?;
    Ok(buf)
}

pub struct NpzBuilder {
    zw: ZipWriter<File>,
}

impl NpzBuilder {
    pub fn create(path: &Path) -> Result<Self> {
        let f = File::create(path)?;
        Ok(Self { zw: ZipWriter::new(f) })
    }

    fn add_blob(&mut self, name: &str, blob: &[u8]) -> Result<()> {
        let opts = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        self.zw.start_file(format!("{name}.npy"), opts)?;
        self.zw.write_all(blob)?;
        Ok(())
    }

    pub fn add_f64(&mut self, name: &str, shape: &[usize], data: &[f64]) -> Result<()> {
        let blob = npy_bytes::<f64>("<f8", shape, data)?;
        self.add_blob(name, &blob)
    }

    pub fn add_f32(&mut self, name: &str, shape: &[usize], data: &[f32]) -> Result<()> {
        let blob = npy_bytes::<f32>("<f4", shape, data)?;
        self.add_blob(name, &blob)
    }

    pub fn add_i32(&mut self, name: &str, shape: &[usize], data: &[i32]) -> Result<()> {
        let blob = npy_bytes::<i32>("<i4", shape, data)?;
        self.add_blob(name, &blob)
    }

    pub fn add_scalar_f64(&mut self, name: &str, v: f64) -> Result<()> {
        // numpy stores 0-d scalars as shape () — but `np.savez` of a Python
        // float writes a (1,)-shape under the hood; Python access `arr["v"]`
        // returns a 0-d array. We mirror the simpler path: shape `()`.
        let bytes = v.to_le_bytes();
        let mut buf = Vec::with_capacity(128);
        write_npy_to(&mut buf, "<f8", &[], &bytes)?;
        self.add_blob(name, &buf)
    }

    pub fn add_scalar_i32(&mut self, name: &str, v: i32) -> Result<()> {
        let bytes = v.to_le_bytes();
        let mut buf = Vec::with_capacity(128);
        write_npy_to(&mut buf, "<i4", &[], &bytes)?;
        self.add_blob(name, &buf)
    }

    pub fn finish(mut self) -> Result<()> {
        self.zw.finish()?;
        Ok(())
    }
}
