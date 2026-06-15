//! `echtvar bench` subcommand: compare per-variant lookup throughput (and
//! correctness) between the zip reader and the parquet reader.
//!
//! Method:
//!   1. Enumerate every (chrom, pos, ref, alt) the zip archive knows about.
//!   2. Look each one up in the zip reader (`update_expr_values` hot path) and
//!      time it.
//!   3. Look each one up in the parquet reader (chunk-load + binary_search +
//!      column index) and time it, asserting the resulting per-field values
//!      match the zip reader bit-for-bit.
//!
//! This isolates the storage read path from the htslib VCF I/O so the timing
//! reflects the format change, not record parsing.

use std::collections::HashMap;
use std::time::Instant;

use echtvar_lib::arrowfmt::{ArrowReaderEchtvar, ChunkData};
use echtvar_lib::echtvar::{EchtVars, Value, Variant};
use echtvar_lib::fields;
use echtvar_lib::var32;

/// Minimal Variant impl so we can drive the zip reader's `update_expr_values`.
struct BenchVar {
    chrom: String,
    pos: u32,
    r: Vec<u8>,
    a: Vec<u8>,
}

impl Variant for BenchVar {
    fn chrom(&self) -> String {
        self.chrom.clone()
    }
    fn rid(&self) -> i32 {
        // single synthetic rid; the zip reader keys cache on rid+chunk, but we
        // drive lookups grouped by chrom so this is fine for a bench.
        0
    }
    fn position(&self) -> u32 {
        self.pos
    }
    fn alleles(&self) -> Vec<&[u8]> {
        vec![&self.r, &self.a]
    }
}

/// Resolve the per-field u32 values for one variant against an already-loaded
/// parquet chunk. Mirrors `EchtVars::update_expr_values`' index resolution:
/// short variants binary_search the var32 keys; long variants match by
/// (position, ref, alt) among the chunk's long entries.
fn parquet_lookup(
    chunk: &ChunkData,
    long_index: &HashMap<(u32, Vec<u8>, Vec<u8>), usize>,
    flds: &[fields::Field],
    pos: u32,
    r: &[u8],
    a: &[u8],
    out: &mut [u32],
    warn: &mut i32,
) -> bool {
    let idx = if r.len() + a.len() <= var32::MAX_COMBINED_LEN {
        let enc = var32::encode(pos, r, a, warn);
        chunk.var32s.binary_search(&enc).ok()
    } else {
        // long variant: O(1) lookup via the prebuilt (pos,ref,alt)->ordinal map.
        long_index.get(&(pos, r.to_vec(), a.to_vec())).copied()
    };
    match idx {
        Some(i) => {
            for (fi, _f) in flds.iter().enumerate() {
                out[fi] = chunk.values[fi][i];
            }
            true
        }
        None => false,
    }
}

pub fn bench_main(zpath: &str, ppath: &str) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("[bench] enumerating variants from {}", zpath);
    let mut e = EchtVars::open(zpath);
    let all = e.enumerate_all_alleles()?;
    eprintln!("[bench] {} variants to look up", all.len());

    let n_fields = e.fields.len();

    // ---- zip reader timing ----
    let mut expr_values = vec![0.0f64; n_fields];
    // capture the zip reader's resolved u32 per variant for correctness check.
    let mut zip_vals: Vec<Vec<u32>> = Vec::with_capacity(all.len());
    let t0 = Instant::now();
    for (chrom, pos, r, a) in &all {
        let mut v = BenchVar {
            chrom: chrom.clone(),
            pos: *pos,
            r: r.clone(),
            a: a.clone(),
        };
        e.update_expr_values(&mut v, &mut expr_values);
        // pull the resolved raw values back out of evalues.
        let row: Vec<u32> = e
            .fields
            .iter()
            .map(|f| match e.evalues[f.values_i] {
                Value::Int(i) => i as u32,
                Value::Float(fl) => fl.to_bits(),
            })
            .collect();
        zip_vals.push(row);
    }
    let zip_ms = t0.elapsed().as_millis().max(1);
    eprintln!(
        "[bench] zip   : {} lookups in {} ms ({} /sec)",
        all.len(),
        zip_ms,
        1000 * all.len() as u128 / zip_ms
    );

    // ---- parquet reader timing ----
    let pr = ArrowReaderEchtvar::open(ppath)?;
    let flds = pr.fields.clone();

    let mut warn = 0i32;
    let mut out = vec![0u32; n_fields];
    let mut mism = 0u64;
    let mut hits = 0u64;

    // process grouped by chunk so we load each row group once (matches the zip
    // reader's chunk cache behaviour).
    let t1 = Instant::now();
    let mut cur: Option<(String, u32, Option<ChunkData>)> = None;
    // prebuilt long-variant index for the currently loaded chunk.
    let mut long_index: HashMap<(u32, Vec<u8>, Vec<u8>), usize> = HashMap::new();
    for (i, (chrom, pos, r, a)) in all.iter().enumerate() {
        let chunk_id = pos >> 20;
        let need_reload = match &cur {
            Some((c, ch, _)) => c != chrom || *ch != chunk_id,
            None => true,
        };
        if need_reload {
            let cd = pr.load_chunk(chrom, chunk_id)?;
            long_index.clear();
            if let Some(ref c) = cd {
                for (ord, (lpos, lref, lalt)) in c.longs.iter() {
                    long_index.insert((*lpos, lref.clone(), lalt.clone()), *ord);
                }
            }
            cur = Some((chrom.clone(), chunk_id, cd));
        }
        let chunk = cur.as_ref().unwrap().2.as_ref();
        let found = match chunk {
            Some(cd) => parquet_lookup(cd, &long_index, &flds, *pos, r, a, &mut out, &mut warn),
            None => false,
        };
        if found {
            hits += 1;
            // Compare decoded-to-decoded: zip_vals[i] already holds the zip
            // reader's *decoded* value (as u32 bits). Decode the parquet raw
            // value the same way the zip reader does, then compare. This makes
            // the missing-sentinel and zigzag/multiplier handling symmetric.
            for fi in 0..n_fields {
                let zip_decoded = zip_vals[i][fi];
                let pq_decoded = decode_like_zip(&flds[fi], out[fi]);
                if pq_decoded != zip_decoded {
                    mism += 1;
                    if mism <= 5 {
                        eprintln!(
                            "[bench] MISMATCH {}:{} {}/{} field {} parquet_decoded={} zip_decoded={} (raw={})",
                            chrom,
                            pos + 1,
                            String::from_utf8_lossy(r),
                            String::from_utf8_lossy(a),
                            flds[fi].alias,
                            pq_decoded,
                            zip_decoded,
                            out[fi]
                        );
                    }
                }
            }
        }
    }
    let pq_ms = t1.elapsed().as_millis().max(1);
    eprintln!(
        "[bench] parquet: {} lookups in {} ms ({} /sec), {} hits, {} field-mismatches",
        all.len(),
        pq_ms,
        1000 * all.len() as u128 / pq_ms,
        hits,
        mism
    );

    if mism == 0 {
        eprintln!("[bench] OK: parquet field values match zip on all hits");
    } else {
        eprintln!("[bench] WARNING: {} field-value mismatches", mism);
    }

    Ok(())
}

/// Decode a raw stored u32 exactly as the zip reader's `get_int_value` /
/// `get_float_value` would, returning the value in the same u32-bits form that
/// the bench captures from `EchtVars::evalues`:
///   - Integer/Categorical/Flag: decoded i32 cast to u32
///   - Float: f32::to_bits of the decoded float
/// This includes the u32::MAX missing sentinel and the special 0x7F800001
/// (VCF-missing) float case, so the comparison is decoded-to-decoded and
/// symmetric across the two readers.
fn decode_like_zip(fld: &fields::Field, raw: u32) -> u32 {
    use echtvar_lib::zigzag;
    use ieee754::Ieee754;
    match fld.ftype {
        fields::FieldType::Float => {
            let f: f32 = if raw == u32::MAX {
                if fld.missing_value == 0x7F800001 {
                    Ieee754::from_bits(0x7F800001u32)
                } else {
                    fld.missing_value as f32
                }
            } else if fld.zigzag {
                (zigzag::decode(raw) as f32) / (fld.multiplier as f32)
            } else {
                (raw as f32) / (fld.multiplier as f32)
            };
            f.to_bits()
        }
        _ => {
            let i: i32 = if raw == u32::MAX {
                fld.missing_value
            } else if fld.zigzag {
                zigzag::decode(raw)
            } else {
                raw as i32
            };
            i as u32
        }
    }
}
