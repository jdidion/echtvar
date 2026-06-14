use crate::fields;
use crate::kmer16;
use crate::var32;
use crate::zigzag;
use bincode::Options;
use rust_htslib::bcf;
use std::io::prelude::*;
use std::{fs, io, str};

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::BufReader;

use stream_vbyte::decode::decode;
use stream_vbyte::scalar::Scalar;

// Scalar decoder on every target. Was `Ssse3` on x86_64 behind a
// feature-gated stream-vbyte dep, but that dep required
// `#![feature(portable_simd)]` (nightly-only) on downstream consumers.
// Decode is not on the hot path for typical annotation workloads.
type StreamVbyteDecoder = Scalar;

use ieee754::Ieee754;

#[derive(Debug, Clone, Copy)]
pub enum Value {
    Int(i32),
    Float(f32),
}

impl Value {
    pub fn value(self) -> f64 {
        match self {
            Value::Int(i) => i as f64,
            Value::Float(f) => f as f64,
        }
    }
}

#[derive(Debug)]
pub struct EchtVars {
    pub zip: zip::ZipArchive<std::fs::File>,
    pub chrom: String,
    last_rid: i32,
    pub start: u32,
    pub var32s: Vec<u32>,
    pub longs: Vec<var32::LongVariant>,
    // the values for a chunk are stored in values.
    pub values: Vec<Vec<u32>>,

    // for storing values used by fasteval
    pub evalues: Vec<Value>,
    // values.len() == fields.len() and fields[i] indicates how we
    // handle values[i]
    pub fields: Vec<fields::Field>,
    buffer: Vec<u8>,

    warn: i32,

    // lookup for categorical fields.
    // these will be empty for non-categorical.
    pub strings: Vec<Vec<std::string::String>>,
}

pub trait Variant {
    fn chrom(&self) -> std::string::String;
    fn rid(&self) -> i32;
    fn position(&self) -> u32;
    fn alleles(&self) -> Vec<&[u8]>;
}

#[inline]
pub fn strip_chr(chrom: std::string::String) -> std::string::String {
    if chrom.len() < 4 {
        return chrom;
    }
    let bchrom = chrom.as_bytes();
    if bchrom[0] as char == 'c' && bchrom[1] as char == 'h' && bchrom[2] as char == 'r' {
        return chrom[3..].to_string();
    }
    chrom
}

#[inline]
pub fn bstrip_chr(chrom: &str) -> &str {
    if chrom.len() < 4 {
        return chrom;
    }
    let bchrom = chrom.as_bytes();
    if bchrom[0] as char == 'c' && bchrom[1] as char == 'h' && bchrom[2] as char == 'r' {
        return &chrom[3..];
    }
    chrom
}

#[cfg(test)]
mod chr_tests {
    use super::{bstrip_chr, strip_chr};

    #[test]
    fn test_strip_chr_prefix() {
        assert_eq!(strip_chr("chr1".to_string()), "1");
        assert_eq!(strip_chr("chr22".to_string()), "22");
        assert_eq!(strip_chr("chrX".to_string()), "X");
    }

    #[test]
    fn test_strip_chr_no_prefix() {
        assert_eq!(strip_chr("1".to_string()), "1");
        assert_eq!(strip_chr("22".to_string()), "22");
        assert_eq!(strip_chr("chr".to_string()), "chr"); // len < 4
        assert_eq!(strip_chr("abc".to_string()), "abc");
    }

    #[test]
    fn test_bstrip_chr_prefix() {
        assert_eq!(bstrip_chr("chr1"), "1");
        assert_eq!(bstrip_chr("chr22"), "22");
    }

    #[test]
    fn test_bstrip_chr_no_prefix() {
        assert_eq!(bstrip_chr("1"), "1");
        assert_eq!(bstrip_chr("chr"), "chr");
    }
}

impl Variant for bcf::record::Record {
    fn chrom(&self) -> std::string::String {
        let rid = self.rid().unwrap();
        let n: &[u8] = self.header().rid2name(rid).unwrap();
        str::from_utf8(n).unwrap().to_string()
    }

    fn rid(&self) -> i32 {
        self.rid().unwrap() as i32
    }

    fn position(&self) -> u32 {
        self.pos() as u32
    }

    fn alleles(&self) -> Vec<&[u8]> {
        self.alleles()
    }
}

impl EchtVars {
    pub fn open(path: &str) -> Self {
        let ep = std::path::Path::new(path);
        let file = fs::File::open(ep)
            .unwrap_or_else(|e| panic!("error accessing zip file: \"{}\" ({})", path, e));
        let mut result = EchtVars {
            zip: zip::ZipArchive::new(file)
                .unwrap_or_else(|e| panic!("error opening: \"{}\" as zip file ({})", path, e)),
            chrom: "".to_string(),
            last_rid: -1,
            start: u32::MAX,
            var32s: vec![],
            longs: vec![],
            values: vec![],
            evalues: vec![],
            fields: vec![],
            buffer: vec![],
            strings: vec![],
            warn: 0,
        };

        {
            let mut fc = result
                .zip
                .by_name("echtvar/config.json")
                .expect("unable to open echtvar/config.json");
            let mut contents = String::new();
            fc.read_to_string(&mut contents)
                .expect("eror reading config.json");
            drop(fc);
            let mut flds: Vec<fields::Field> = json5::from_str(&contents).unwrap();
            for fld in flds.iter_mut() {
                fld.values_i = result.fields.len();
                if fld.ftype == fields::FieldType::Categorical {
                    // read in the strings for this field. replace ';' with ',' to handle the filter field.
                    let fname = format!("echtvar/strings/{}.txt", fld.alias);
                    let fh = result
                        .zip
                        .by_name(&fname)
                        .expect("error opening strings file");
                    result.strings.push(
                        BufReader::new(fh)
                            .lines()
                            .map(|l| l.unwrap().replace(';', ","))
                            .collect(),
                    );
                    // update missing value to be the index of the missing_string
                    let strings_len = result.strings[result.strings.len() - 1].len();
                    fld.missing_value = result.strings[result.strings.len() - 1]
                        .iter()
                        .position(|s| s == &fld.missing_string)
                        .unwrap_or(strings_len) as i32;
                    // if it wasn't in the list, add it.
                    if fld.missing_value == strings_len as i32 {
                        let rl = result.strings.len() - 1;
                        result.strings[rl].push(fld.missing_string.clone());
                    }
                } else {
                    result.strings.push(Vec::new());
                }
                let f = fld.clone();
                result.fields.push(f);
            }
            result.values.resize(result.fields.len(), vec![]);
            result.evalues.resize(result.fields.len(), Value::Int(0));
        }
        eprintln!("fields: {:?}", result.fields);
        result
    }

    pub fn update_header(self: &mut EchtVars, header: &mut bcf::header::Header, path: &str) {
        for e in &self.fields {
            header.push_record(
                format!(
                    "##INFO=<ID={},Number={},Type={},Description=\"{}\">",
                    e.alias,
                    if e.ftype == fields::FieldType::Flag {
                        "0"
                    } else if ["A", "R", "G"].contains(&e.number.as_str()) {
                        "1"
                    } else {
                        &e.number
                    },
                    if e.ftype == fields::FieldType::Flag {
                        "Flag"
                    } else if e.ftype == fields::FieldType::Integer {
                        "Integer"
                    } else if e.ftype == fields::FieldType::Categorical {
                        "String"
                    } else {
                        "Float"
                    },
                    if e.description == fields::default_description_string() {
                        format!("added by echtvar from {}", path)
                    } else {
                        format!("added by echtvar {}", e.description.to_string())
                    }
                )
                .as_bytes(),
            );
        }
    }
    pub fn add_cmd_header(
        header: &mut bcf::header::Header,
        vpath: &str,
        opath: &str,
        include_expr: &Option<&str>,
        epaths: Vec<&str>,
    ) {
        header.push_record(
            format!(
                "##echtvar_annoCommand=anno -i {:?} {} {} -e {:?}",
                include_expr,
                vpath,
                opath,
                epaths.join(" -e ")
            )
            .as_bytes(),
        );
    }

    #[inline(always)]
    pub fn set_position(
        self: &mut EchtVars,
        rid: i32,
        chromosome: String,
        position: u32,
    ) -> io::Result<()> {
        if rid == self.last_rid && position >> 20 == self.start >> 20 {
            return Ok(());
        }
        self.last_rid = rid;
        self.start = position >> 20 << 20; // round to 20 bits.
        self.chrom = strip_chr(chromosome);
        let base_path = format!("echtvar/{}/{}", self.chrom, position >> 20);

        for fi in self.fields.iter_mut() {
            // RUST-TODO: use .fill function. problems with double borrow.
            let path = format!("{}/{}.bin", base_path, fi.alias);
            let rzip = self.zip.by_name(&path);
            match rzip {
                Ok(mut iz) => {
                    let n = iz.read_u32::<LittleEndian>()? as usize;
                    self.buffer
                        .resize(iz.size() as usize - std::mem::size_of::<u32>(), 0x0);
                    iz.read_exact(&mut self.buffer)?;
                    self.values[fi.values_i].resize(n, 0x0);
                    // TODO: use skip to first position.
                    let bytes_decoded =
                        decode::<StreamVbyteDecoder>(&self.buffer, n, &mut self.values[fi.values_i]);

                    if bytes_decoded != self.buffer.len() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "didn't read expected number of values from zip",
                        ));
                    }
                }
                _ => {
                    // if we don't find this file we set everything to empty.
                    self.values[fi.values_i].clear();
                    self.var32s.clear();
                }
            }
        }

        if !self.values[0].is_empty() {
            let path = format!("{}/var32.bin", base_path);
            let mut iz = self.zip.by_name(&path)?;
            let n = iz.read_u32::<LittleEndian>()? as usize;
            //eprintln!("n:{}", n);
            self.buffer
                .resize(iz.size() as usize - std::mem::size_of::<u32>(), 0x0);
            iz.read_exact(&mut self.buffer)?;

            self.var32s.resize(n, 0x0);
            let bytes_decoded = decode::<StreamVbyteDecoder>(&self.buffer, n, &mut self.var32s);
            // cumsum https://users.rust-lang.org/t/inplace-cumulative-sum-using-iterator/56532/3
            self.var32s.iter_mut().fold(0, |acc, x| {
                *x += acc;
                *x
            });

            if bytes_decoded != self.buffer.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "didn't read expected number of values from zip",
                ));
            }
        }

        if !self.var32s.is_empty() {
            let long_path = format!("{}/too-long-for-var32.enc", base_path);
            let mut iz = self.zip.by_name(&long_path)?;
            self.buffer.clear();
            iz.read_to_end(&mut self.buffer)?;
            self.longs = bincode::DefaultOptions::new()
                .deserialize(&self.buffer)
                .expect("error decoding long variants");
        } else {
            self.longs.clear();
        }

        Ok(())
    }

    #[inline]
    fn get_int_value(self: &EchtVars, fld: &fields::Field, idx: usize) -> i32 {
        let v: u32 = self.values[fld.values_i][idx];
        if v == u32::MAX {
            fld.missing_value
        } else if fld.zigzag {
            zigzag::decode(v)
        } else {
            v as i32
        }
    }

    #[inline]
    fn get_float_value(self: &EchtVars, fld: &fields::Field, idx: usize) -> f32 {
        let v: u32 = self.values[fld.values_i][idx];
        if v == u32::MAX {
            if fld.missing_value == 0x7F800001 {
                Ieee754::from_bits(0x7F800001)
            } else {
                fld.missing_value as f32
            }
        } else if fld.zigzag {
            (zigzag::decode(v) as f32) / (fld.multiplier as f32)
        } else {
            (v as f32) / (fld.multiplier as f32)
        }
    }

    pub fn set_position_by_name(&mut self, chrom: &str, position: u32) -> io::Result<()> {
        let stripped = strip_chr(chrom.to_string());
        if self.chrom == stripped && position >> 20 == self.start >> 20 {
            return Ok(());
        }
        self.last_rid = i32::MIN;
        self.set_position(i32::MIN + 1, chrom.to_string(), position)
    }

    /// Return all (ref, alt) allele pairs at the given 0-based position within the
    /// currently loaded chunk. This is used by the BED/tab position-scan mode where
    /// the input has no REF/ALT columns: we enumerate every variant the echtvar file
    /// knows about at that position so each can be annotated as a separate output row.
    /// The existing VCF path doesn't need this because the VCF itself provides the alleles.
    pub fn variants_at_position(&self, pos: u32) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut results = Vec::new();

        // Search var32s: position is stored in lower 20 bits
        let pos_in_chunk = pos & 0xFFFFF;
        let min_val = pos_in_chunk << 12;
        let max_val = min_val | 0xFFF;

        let lo = self.var32s.partition_point(|&v| v < min_val);
        let mut i = lo;
        while i < self.var32s.len() && self.var32s[i] <= max_val {
            if let Some(alleles) = var32::decode_to_alleles(self.var32s[i]) {
                results.push(alleles);
            }
            i += 1;
        }

        // Search long variants: sorted by position, so use binary search
        let lo = self.longs.partition_point(|l| l.position < pos);
        for l in self.longs[lo..].iter().take_while(|l| l.position == pos) {
            let (ref_allele, alt_allele) = kmer16::decode_var(&l.sequence);
            results.push((ref_allele, alt_allele));
        }

        results
    }

    /// Enumerate every `(chrom, position_0based, ref_allele, alt_allele)` tuple
    /// stored in the archive.
    ///
    /// Walks the zip file listing to discover every `(chrom, chunk_id)` pair
    /// (chunk layout is `echtvar/{chrom_stripped}/{chunk_id}/...`), loads each
    /// chunk via [`Self::set_position`], and decodes both the `var32`-packed
    /// short variants and the `too-long-for-var32` long variants.
    ///
    /// `chrom` in the returned tuples is the stripped contig name (no `chr`
    /// prefix) — matches echtvar's on-disk convention. Callers that need the
    /// original `chr` prefix should attach it themselves.
    ///
    /// This is the region-enumeration helper that powers force-call injection
    /// under `--variants`: a caller that wants every panel allele overlapping
    /// a region materializes this vec once, groups by contig, and filters by
    /// position at region-query time. Memory cost scales with panel size
    /// (~40 bytes per tuple); a 34k-variant A45 panel is ~1.4 MB.
    pub fn enumerate_all_alleles(&mut self) -> io::Result<Vec<(String, u32, Vec<u8>, Vec<u8>)>> {
        // Collect unique (chrom, chunk_id) pairs by scanning zip entry names.
        // Layout: `echtvar/{chrom_stripped}/{chunk_id}/var32.bin`. We key on
        // var32.bin to avoid double-counting per-field value files.
        let mut chunks: Vec<(String, u32)> = Vec::new();
        {
            for i in 0..self.zip.len() {
                let entry = self.zip.by_index(i)?;
                let name = entry.name().to_string();
                // Accept only the var32.bin sentinel to enumerate chunks once.
                if let Some(rest) = name.strip_prefix("echtvar/") {
                    if let Some(stripped) = rest.strip_suffix("/var32.bin") {
                        let mut parts = stripped.splitn(2, '/');
                        let chrom = match parts.next() {
                            Some(c) if !c.is_empty() => c.to_string(),
                            _ => continue,
                        };
                        let chunk_str = match parts.next() {
                            Some(c) => c,
                            None => continue,
                        };
                        if let Ok(chunk_id) = chunk_str.parse::<u32>() {
                            chunks.push((chrom, chunk_id));
                        }
                    }
                }
            }
        }
        // Sort so output order is stable (by contig name, then chunk id).
        chunks.sort();

        let mut out: Vec<(String, u32, Vec<u8>, Vec<u8>)> = Vec::new();
        // Fresh synthetic rid per chunk-group so `set_position`'s cache check
        // always reloads. Using i32::MIN+k as a non-colliding sentinel stream.
        let mut synthetic_rid: i32 = i32::MIN;
        let mut last_chrom: Option<String> = None;
        for (chrom, chunk_id) in &chunks {
            if last_chrom.as_deref() != Some(chrom.as_str()) {
                synthetic_rid = synthetic_rid.saturating_add(1);
                last_chrom = Some(chrom.clone());
            }
            let chunk_base: u32 = chunk_id << 20;
            // `set_position` uses `position >> 20` as the chunk id, so hand it
            // any position inside the chunk.
            self.set_position(synthetic_rid, chrom.clone(), chunk_base)?;

            // Short variants: var32s holds pos_in_chunk<<12 | enc, cumsum'd.
            for &v in &self.var32s {
                if let Some((r, a)) = var32::decode_to_alleles(v) {
                    let pos_in_chunk = v >> 12;
                    let pos = chunk_base | pos_in_chunk;
                    out.push((chrom.clone(), pos, r, a));
                }
            }
            // Long variants: each carries its absolute position explicitly.
            for l in &self.longs {
                let (r, a) = kmer16::decode_var(&l.sequence);
                out.push((chrom.clone(), l.position, r, a));
            }
        }
        Ok(out)
    }

    pub fn update_expr_values<T: Variant>(
        self: &mut EchtVars,
        variant: &mut T,
        expr_values: &mut [f64],
    ) {
        let pos = variant.position();
        let rid = variant.rid();
        if rid != self.last_rid || pos >> 20 != self.start >> 20 {
            let chrom = variant.chrom();
            let _ = self.set_position(rid, chrom, pos);
        }

        let alleles = variant.alleles();
        if alleles.len() != 2 {
            panic!(
                "[echtvar] variants must be decomposed before running. got variant with {} alleles at {}:{} ({:?}). see: https://github.com/brentp/echtvar/wiki/decompose",
                alleles.len() - 1,
                variant.chrom(),
                variant.position() + 1,
                variant.alleles()
            );
        }
        let eidx = if alleles[0].len() + alleles[1].len() <= crate::var32::MAX_COMBINED_LEN {
            let enc = var32::encode(pos, alleles[0], alleles[1], &mut self.warn);
            self.var32s.binary_search(&enc)
        } else {
            let l = var32::LongVariant {
                position: pos,
                sequence: kmer16::encode_var(alleles[0], alleles[1]),
                idx: 0,
            };
            let r = self.longs.binary_search(&l);
            match r {
                Ok(idx) => Ok(self.longs[idx].idx as usize),
                Err(_) => Err(0),
            }
        };
        match eidx {
            Ok(idx) => {
                for fld in &self.fields {
                    if fld.ftype == fields::FieldType::Integer
                        || fld.ftype == fields::FieldType::Categorical
                        || fld.ftype == fields::FieldType::Flag
                    {
                        let val = self.get_int_value(fld, idx);
                        self.evalues[fld.values_i] = Value::Int(val);
                        expr_values[fld.values_i] = val as f64
                    } else if fld.ftype == fields::FieldType::Float {
                        let val = self.get_float_value(fld, idx);
                        self.evalues[fld.values_i] = Value::Float(val);
                        expr_values[fld.values_i] = val as f64
                    } else {
                        panic!("not implemented");
                    }
                }
            }
            Err(_) => {
                for fld in &self.fields {
                    if fld.ftype == fields::FieldType::Integer
                        || fld.ftype == fields::FieldType::Categorical
                        || fld.ftype == fields::FieldType::Flag
                    {
                        // for Categorical missing_value has been set to the index of missing_string
                        let val = fld.missing_value;
                        self.evalues[fld.values_i] = Value::Int(val);
                        expr_values[fld.values_i] = val as f64
                    } else if fld.ftype == fields::FieldType::Float {
                        let val = if fld.missing_value == 0x7F800001 {
                            Ieee754::from_bits(0x7F800001)
                        } else {
                            fld.missing_value as f32
                        };
                        self.evalues[fld.values_i] = Value::Float(val);
                        expr_values[fld.values_i] = val as f64
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    //#[test]
    #[allow(dead_code)]
    fn test_read() {
        let mut e = EchtVars::open("ec.zip");
        e.set_position(22, "chr21".to_string(), 5030088).ok();

        assert_eq!(e.fields.len(), 3);
        assert_eq!(e.values[0].len(), 46881);
        assert_eq!(e.values[1].len(), e.var32s.len());

        assert_eq!(e.longs[0].position, 5030185);
    }

    //#[test]
    #[allow(dead_code)]
    fn test_search() {
        let mut e = EchtVars::open("ec.zip");
        e.set_position(22, "chr21".to_string(), 5030088).ok();

        let mut vals = vec![];
        vals.resize(3, 0.0);

        pub struct Var<'a> {
            chrom: std::string::String, //b"chr21"
            pos: u32,                   // 5030087,
            alleles: Vec<&'a [u8]>,     //vec!["C", "T"],
        }

        impl<'a> Variant for Var<'a> {
            fn chrom(&self) -> std::string::String {
                self.chrom.clone()
            }
            fn position(&self) -> u32 {
                self.pos
            }
            fn rid(&self) -> i32 {
                1
            }
            fn alleles(&self) -> Vec<&[u8]> {
                self.alleles.clone()
            }
        }

        let mut variant = Var {
            chrom: "chr21".to_string(),
            pos: 5030087,
            alleles: vec![b"C", b"T"],
        };

        let idx = e.update_expr_values(&mut variant, &mut vals);
        eprintln!("vals:{:?} {:?}", vals, idx);
        assert_eq!(vals[1], 2.0);
    }
}
