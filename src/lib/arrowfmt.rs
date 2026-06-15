//! Experimental Arrow/Parquet on-disk format for echtvar.
//!
//! This is a *container swap*, not a re-encoding. The zip format stores, per
//! 1<<20-base genome chunk:
//!   - `var32.bin`     : stream-vbyte( delta( sorted u32 var32 keys ) )
//!   - `<alias>.bin`   : stream-vbyte( per-field u32 values, in var32 order )
//!   - `too-long-for-var32.enc` : bincode(Vec<LongVariant>)
//!
//! Here we keep the *exact same u32 encoding* (var32 keys, u32::MAX missing
//! sentinel, multiplier-scaled floats, categorical string indices) but put it
//! in a single Parquet file:
//!   - one **row group per (chrom, chunk)**, so chunk lookup = pick a row group
//!   - columns: `chrom` (Utf8), `chunk` (UInt32), `var32` (UInt32, sorted),
//!     one `UInt32` column per field (named by alias),
//!     and long-variant columns (`long_idx`, `long_pos`, `long_ref`,
//!     `long_alt`) that are null for short variants.
//!
//! The annotation hot path is identical to the zip reader: load the row group
//! for `pos >> 20`, `binary_search` the var32 column, index the field columns.
//!
//! Categorical string tables and the field config travel in the Parquet
//! file-level key-value metadata (key `echtvar_config` = the same JSON the zip
//! stores at `echtvar/config.json`, and `echtvar_strings` = a JSON map of
//! alias -> [levels]).

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, BinaryArray, BinaryBuilder, StringArray, UInt32Array, UInt32Builder,
};
use arrow::datatypes::{DataType, Field as ArrowField, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

use crate::fields;
use crate::var32;

/// JSON-config metadata key in the parquet footer.
pub const CONFIG_KEY: &str = "echtvar_config";
/// Categorical string-levels metadata key in the parquet footer.
pub const STRINGS_KEY: &str = "echtvar_strings";

/// Fixed (non-field) column names.
const COL_CHROM: &str = "__chrom";
const COL_CHUNK: &str = "__chunk";
const COL_VAR32: &str = "__var32";
const COL_LONG_POS: &str = "__long_pos";
const COL_LONG_REF: &str = "__long_ref";
const COL_LONG_ALT: &str = "__long_alt";

/// One decoded chunk's worth of data, all parallel to `var32` (which is the
/// sorted key). Long variants are recovered into the same per-row position via
/// their original ordinal so lookup logic matches the zip reader.
pub struct ChunkData {
    pub var32s: Vec<u32>,
    /// values[field_index] is parallel to var32s.
    pub values: Vec<Vec<u32>>,
    /// (position, ref, alt) for long variants in this chunk, keyed by their
    /// ordinal index into var32s.
    pub longs: HashMap<usize, (u32, Vec<u8>, Vec<u8>)>,
}

/// Build the Arrow schema for a given set of fields. Field columns are named by
/// their alias and carry the raw u32 (sentinel-encoded) values.
pub fn build_schema(flds: &[fields::Field]) -> Arc<Schema> {
    let mut cols: Vec<ArrowField> = Vec::with_capacity(flds.len() + 6);
    cols.push(ArrowField::new(COL_CHROM, DataType::Utf8, false));
    cols.push(ArrowField::new(COL_CHUNK, DataType::UInt32, false));
    cols.push(ArrowField::new(COL_VAR32, DataType::UInt32, false));
    for f in flds {
        // field value columns are non-null: missing is the u32::MAX sentinel,
        // exactly as the zip stores it.
        cols.push(ArrowField::new(&f.alias, DataType::UInt32, false));
    }
    // long-variant side columns: null for short variants.
    cols.push(ArrowField::new(COL_LONG_POS, DataType::UInt32, true));
    cols.push(ArrowField::new(COL_LONG_REF, DataType::Binary, true));
    cols.push(ArrowField::new(COL_LONG_ALT, DataType::Binary, true));
    Arc::new(Schema::new(cols))
}

/// Writer for the parquet echtvar format. One [`write_chunk`] call per
/// (chrom, chunk) emits exactly one row group.
pub struct ArrowWriterEchtvar {
    writer: ArrowWriter<std::fs::File>,
    schema: Arc<Schema>,
    n_fields: usize,
}

impl ArrowWriterEchtvar {
    pub fn create(
        path: &str,
        flds: &[fields::Field],
        strings: &[Vec<String>],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let schema = build_schema(flds);

        let config_json = serde_json::to_string(flds)?;
        // strings: alias -> [levels]; only categorical fields have non-empty.
        let mut strings_map: HashMap<&str, &Vec<String>> = HashMap::new();
        for (f, s) in flds.iter().zip(strings.iter()) {
            if !s.is_empty() {
                strings_map.insert(f.alias.as_str(), s);
            }
        }
        let strings_json = serde_json::to_string(&strings_map)?;

        let kv = vec![
            parquet::file::metadata::KeyValue::new(
                CONFIG_KEY.to_string(),
                config_json,
            ),
            parquet::file::metadata::KeyValue::new(
                STRINGS_KEY.to_string(),
                strings_json,
            ),
        ];

        // Per-column encoding. The var32 key column is sorted within each chunk,
        // so delta-pack it (mirrors the zip's delta + stream-vbyte) — this is the
        // single biggest size lever, the keys are ~63% of the file under the
        // default PLAIN encoding. Long-variant positions are also sorted.
        // DELTA_BINARY_PACKED requires the dictionary disabled for that column.
        //
        // zstd level is offline/cold-path, so a high level is cheap; allow an
        // ECHTVAR_ZSTD override for sweeping. long_ref/long_alt are DNA strings
        // with shared prefixes — DELTA_BYTE_ARRAY (incremental) beats PLAIN.
        let zlevel: i32 = std::env::var("ECHTVAR_ZSTD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7);
        let var32_path = ColumnPath::from(COL_VAR32);
        let long_pos_path = ColumnPath::from(COL_LONG_POS);
        let long_ref_path = ColumnPath::from(COL_LONG_REF);
        let long_alt_path = ColumnPath::from(COL_LONG_ALT);
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(zlevel)?))
            // each write() flushes one row group; keep the cap high so a chunk
            // is never split across row groups.
            .set_max_row_group_row_count(Some(1 << 30))
            .set_key_value_metadata(Some(kv))
            .set_column_dictionary_enabled(var32_path.clone(), false)
            .set_column_encoding(var32_path, Encoding::DELTA_BINARY_PACKED)
            .set_column_dictionary_enabled(long_pos_path.clone(), false)
            .set_column_encoding(long_pos_path, Encoding::DELTA_BINARY_PACKED)
            .set_column_dictionary_enabled(long_ref_path.clone(), false)
            .set_column_encoding(long_ref_path, Encoding::DELTA_BYTE_ARRAY)
            .set_column_dictionary_enabled(long_alt_path.clone(), false)
            .set_column_encoding(long_alt_path, Encoding::DELTA_BYTE_ARRAY)
            .build();

        let file = std::fs::File::create(path)?;
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;
        Ok(ArrowWriterEchtvar {
            writer,
            schema,
            n_fields: flds.len(),
        })
    }

    /// Write a single (chrom, chunk) as one row group. `var32s` must be the
    /// sorted keys; `values[i]` parallel to var32s; `longs` maps the var32
    /// ordinal to (abs_position, ref, alt).
    pub fn write_chunk(
        &mut self,
        chrom: &str,
        chunk: u32,
        var32s: &[u32],
        values: &[Vec<u32>],
        longs: &HashMap<usize, (u32, Vec<u8>, Vec<u8>)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let n = var32s.len();
        if n == 0 {
            return Ok(());
        }
        let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(self.n_fields + 6);

        // chrom (constant) -> dict/RLE compresses to ~nothing.
        let chrom_arr = StringArray::from(vec![chrom; n]);
        arrays.push(Arc::new(chrom_arr));
        // chunk (constant)
        let chunk_arr = UInt32Array::from(vec![chunk; n]);
        arrays.push(Arc::new(chunk_arr));
        // var32 keys
        arrays.push(Arc::new(UInt32Array::from(var32s.to_vec())));
        // field value columns
        for v in values.iter() {
            debug_assert_eq!(v.len(), n);
            arrays.push(Arc::new(UInt32Array::from(v.to_vec())));
        }
        // long-variant side columns
        let mut long_pos = UInt32Builder::new();
        let mut long_ref = BinaryBuilder::new();
        let mut long_alt = BinaryBuilder::new();
        for i in 0..n {
            match longs.get(&i) {
                Some((pos, r, a)) => {
                    long_pos.append_value(*pos);
                    long_ref.append_value(r);
                    long_alt.append_value(a);
                }
                None => {
                    long_pos.append_null();
                    long_ref.append_null();
                    long_alt.append_null();
                }
            }
        }
        arrays.push(Arc::new(long_pos.finish()));
        arrays.push(Arc::new(long_ref.finish()));
        arrays.push(Arc::new(long_alt.finish()));

        let batch = RecordBatch::try_new(self.schema.clone(), arrays)?;
        self.writer.write(&batch)?;
        // force a row-group boundary at each chunk.
        self.writer.flush()?;
        Ok(())
    }

    pub fn close(self) -> Result<(), Box<dyn std::error::Error>> {
        self.writer.close()?;
        Ok(())
    }
}

/// Reader for the parquet echtvar format. Holds the config/strings parsed from
/// footer metadata and a `(chrom, chunk) -> row_group_index` map built from
/// row-group statistics so chunk lookup never scans data.
pub struct ArrowReaderEchtvar {
    path: String,
    pub fields: Vec<fields::Field>,
    pub strings: Vec<Vec<String>>,
    /// (chrom_stripped, chunk_id) -> row group index.
    rg_index: HashMap<(String, u32), usize>,
    /// arrow column index for each field's value column (parallel to fields).
    field_col_idx: Vec<usize>,
    var32_col_idx: usize,
    long_pos_col_idx: usize,
    long_ref_col_idx: usize,
    long_alt_col_idx: usize,
    /// Parsed footer metadata, loaded once and reused for every chunk read so
    /// we don't reparse the footer per row group (918x on gnomAD).
    arrow_meta: ArrowReaderMetadata,
}

impl ArrowReaderEchtvar {
    pub fn open(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        // Parse the footer once; reuse for every chunk read.
        let arrow_meta = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())?;
        let meta = arrow_meta.metadata().clone();
        let file_meta = meta.file_metadata();

        // parse config + strings from footer kv metadata.
        let mut config_json: Option<String> = None;
        let mut strings_json: Option<String> = None;
        if let Some(kvs) = file_meta.key_value_metadata() {
            for kv in kvs {
                if kv.key == CONFIG_KEY {
                    config_json = kv.value.clone();
                } else if kv.key == STRINGS_KEY {
                    strings_json = kv.value.clone();
                }
            }
        }
        let config_json =
            config_json.ok_or("parquet file missing echtvar_config metadata")?;
        let mut flds: Vec<fields::Field> = serde_json::from_str(&config_json)?;

        let strings_map: HashMap<String, Vec<String>> = match strings_json {
            Some(s) => serde_json::from_str(&s)?,
            None => HashMap::new(),
        };

        // mirror EchtVars::open: assign values_i, attach string tables, and fix
        // the categorical missing_value to the index of missing_string.
        let mut strings: Vec<Vec<String>> = Vec::with_capacity(flds.len());
        for (i, fld) in flds.iter_mut().enumerate() {
            fld.values_i = i;
            if fld.ftype == fields::FieldType::Categorical {
                let mut levels = strings_map.get(&fld.alias).cloned().unwrap_or_default();
                let strings_len = levels.len();
                fld.missing_value = levels
                    .iter()
                    .position(|s| s == &fld.missing_string)
                    .unwrap_or(strings_len) as i32;
                if fld.missing_value == strings_len as i32 {
                    levels.push(fld.missing_string.clone());
                }
                strings.push(levels);
            } else {
                strings.push(Vec::new());
            }
        }

        // resolve arrow column indices by name.
        let schema = arrow_meta.schema();
        let col = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
            schema
                .index_of(name)
                .map_err(|_| format!("parquet schema missing column {}", name).into())
        };
        let var32_col_idx = col(COL_VAR32)?;
        let long_pos_col_idx = col(COL_LONG_POS)?;
        let long_ref_col_idx = col(COL_LONG_REF)?;
        let long_alt_col_idx = col(COL_LONG_ALT)?;
        let mut field_col_idx = Vec::with_capacity(flds.len());
        for f in &flds {
            field_col_idx.push(col(&f.alias)?);
        }

        // build (chrom, chunk) -> row group from per-row-group statistics on the
        // constant chrom/chunk columns. We look up the columns by their position
        // in the *parquet* schema; chrom is col 0, chunk col 1 by construction.
        let chrom_pq = col(COL_CHROM)?;
        let chunk_pq = col(COL_CHUNK)?;
        let mut rg_index = HashMap::new();
        for (rg_i, rg) in meta.row_groups().iter().enumerate() {
            let chrom_stats = rg.column(chrom_pq).statistics();
            let chunk_stats = rg.column(chunk_pq).statistics();
            let chrom = match chrom_stats {
                Some(parquet::file::statistics::Statistics::ByteArray(s)) => s
                    .min_opt()
                    .map(|b| String::from_utf8_lossy(b.data()).to_string()),
                _ => None,
            };
            let chunk = match chunk_stats {
                Some(parquet::file::statistics::Statistics::Int32(s)) => {
                    s.min_opt().map(|v| *v as u32)
                }
                _ => None,
            };
            if let (Some(c), Some(ch)) = (chrom, chunk) {
                rg_index.insert((c, ch), rg_i);
            }
        }

        Ok(ArrowReaderEchtvar {
            path: path.to_string(),
            fields: flds,
            strings,
            rg_index,
            field_col_idx,
            var32_col_idx,
            long_pos_col_idx,
            long_ref_col_idx,
            long_alt_col_idx,
            arrow_meta,
        })
    }

    /// Distinct (chrom, chunk) pairs known to this file (for enumeration / bench).
    pub fn chunks(&self) -> Vec<(String, u32)> {
        let mut v: Vec<(String, u32)> = self.rg_index.keys().cloned().collect();
        v.sort();
        v
    }

    /// Load the full chunk for (chrom_stripped, chunk_id). Returns None if that
    /// chunk isn't present (the annotation path treats this as "all missing").
    pub fn load_chunk(
        &self,
        chrom: &str,
        chunk: u32,
    ) -> Result<Option<ChunkData>, Box<dyn std::error::Error>> {
        let rg_i = match self.rg_index.get(&(chrom.to_string(), chunk)) {
            Some(i) => *i,
            None => return Ok(None),
        };
        self.load_row_group(rg_i).map(Some)
    }

    /// Number of row groups (== number of chunks) in the file, in file order.
    pub fn num_row_groups(&self) -> usize {
        self.rg_index.len()
    }

    /// Load a row group by its index (for sequential iteration over the whole
    /// file), decoding *all* field columns.
    pub fn load_row_group(
        &self,
        rg_i: usize,
    ) -> Result<ChunkData, Box<dyn std::error::Error>> {
        self.load_row_group_projected(rg_i, None)
    }

    /// Load a row group, optionally decoding only a subset of the field columns.
    /// `projection` is a set of field indices (into `self.fields`); `None` means
    /// all fields. The var32 key column and the long-variant columns are always
    /// decoded (they're needed to resolve any lookup). Field columns not in the
    /// projection are returned as empty `values[fi]` vecs.
    ///
    /// Reuses the footer metadata parsed in `open` (via `new_with_metadata`), so
    /// no per-chunk footer reparse. A fresh file handle is opened per call (a
    /// cheap syscall relative to decoding a ~14k-row group).
    pub fn load_row_group_projected(
        &self,
        rg_i: usize,
        projection: Option<&[usize]>,
    ) -> Result<ChunkData, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(&self.path)?;
        let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(
            file,
            self.arrow_meta.clone(),
        );

        // Build a projection mask over leaf columns. Always include var32 +
        // long_* ; include the requested field leaves (or all fields).
        let parquet_schema = builder.parquet_schema();
        let mut leaves: Vec<usize> = vec![
            self.var32_col_idx,
            self.long_pos_col_idx,
            self.long_ref_col_idx,
            self.long_alt_col_idx,
        ];
        let decode_fields: Vec<usize> = match projection {
            Some(p) => p.to_vec(),
            None => (0..self.fields.len()).collect(),
        };
        for &fi in &decode_fields {
            leaves.push(self.field_col_idx[fi]);
        }
        // Our schema is flat (no nesting), so arrow column index == leaf index.
        let mask = ProjectionMask::leaves(parquet_schema, leaves.iter().copied());

        let rg_meta_rows = builder.metadata().row_group(rg_i).num_rows() as usize;
        let reader = builder
            .with_row_groups(vec![rg_i])
            .with_projection(mask)
            .with_batch_size(rg_meta_rows.max(1))
            .build()?;

        let mut var32s: Vec<u32> = Vec::new();
        let mut values: Vec<Vec<u32>> = (0..self.fields.len()).map(|_| Vec::new()).collect();
        let mut longs: HashMap<usize, (u32, Vec<u8>, Vec<u8>)> = HashMap::new();

        for batch in reader {
            let batch = batch?;
            let base = var32s.len();
            // With a projection, the batch only contains the projected columns,
            // in schema order. Resolve each needed column by NAME so absolute
            // indices don't matter.
            let schema = batch.schema();
            let by_name = |b: &arrow::record_batch::RecordBatch, name: &str| {
                schema.index_of(name).ok().map(|i| b.column(i).clone())
            };

            let v_col = by_name(&batch, COL_VAR32).ok_or("var32 column missing")?;
            let v = v_col
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or("var32 column type mismatch")?;
            var32s.extend(v.values().iter().copied());

            for &fi in &decode_fields {
                let name = self.fields[fi].alias.as_str();
                let col = by_name(&batch, name).ok_or("field column missing")?;
                let col = col
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .ok_or("field column type mismatch")?;
                values[fi].extend(col.values().iter().copied());
            }

            let lpos_col = by_name(&batch, COL_LONG_POS).ok_or("long_pos missing")?;
            let lref_col = by_name(&batch, COL_LONG_REF).ok_or("long_ref missing")?;
            let lalt_col = by_name(&batch, COL_LONG_ALT).ok_or("long_alt missing")?;
            let lpos = lpos_col
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or("long_pos type mismatch")?;
            let lref = lref_col
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or("long_ref type mismatch")?;
            let lalt = lalt_col
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or("long_alt type mismatch")?;
            for row in 0..batch.num_rows() {
                if lpos.is_valid(row) && lref.is_valid(row) {
                    longs.insert(
                        base + row,
                        (
                            lpos.value(row),
                            lref.value(row).to_vec(),
                            lalt.value(row).to_vec(),
                        ),
                    );
                }
            }
        }

        Ok(ChunkData {
            var32s,
            values,
            longs,
        })
    }
}

/// Decode a long variant's stored ref/alt against an input variant. (kept here
/// so the annotation path can match the zip reader's allele handling without
/// the kmer16 round-trip — the parquet format stores ref/alt as raw bytes.)
pub fn long_matches(stored_ref: &[u8], stored_alt: &[u8], in_ref: &[u8], in_alt: &[u8]) -> bool {
    stored_ref == in_ref && stored_alt == in_alt
}

/// Convenience: encode an input (pos, ref, alt) into the var32 key used to
/// search a chunk's var32s (short variants only).
pub fn encode_short(pos: u32, ref_allele: &[u8], alt_allele: &[u8], warn: &mut i32) -> u32 {
    var32::encode(pos, ref_allele, alt_allele, warn)
}
