//! `echtvar arrow` subcommand: convert an existing zip-format echtvar archive
//! to the experimental Parquet format, with no VCF re-parse.
//!
//! We reuse the zip reader's chunk-walking: for every (chrom, chunk) present in
//! the archive we call `EchtVars::set_position`, which decodes that chunk's
//! `var32s`, per-field `values`, and `longs` exactly as annotation would. We
//! then hand those already-decoded u32 arrays straight to the Parquet writer.

use std::collections::HashMap;

use echtvar_lib::arrowfmt::ArrowWriterEchtvar;
use echtvar_lib::echtvar::EchtVars;
use echtvar_lib::kmer16;

/// Discover every (chrom_stripped, chunk_id) in the zip by scanning entry names
/// for the `var32.bin` sentinel — same approach as `enumerate_all_alleles`.
fn discover_chunks(e: &mut EchtVars) -> std::io::Result<Vec<(String, u32)>> {
    let mut chunks: Vec<(String, u32)> = Vec::new();
    for i in 0..e.zip.len() {
        let entry = e.zip.by_index(i)?;
        let name = entry.name().to_string();
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
    chunks.sort();
    Ok(chunks)
}

/// Print per-column compressed/uncompressed sizes and encodings for a parquet
/// file, summed across all row groups. Diagnostic for the size investigation.
pub fn pqstat_main(ppath: &str) -> Result<(), Box<dyn std::error::Error>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(ppath)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let meta = builder.metadata();
    let nrg = meta.num_row_groups();
    let ncols = meta.file_metadata().schema_descr().num_columns();

    let mut names: Vec<String> = Vec::with_capacity(ncols);
    for c in 0..ncols {
        names.push(meta.file_metadata().schema_descr().column(c).path().string());
    }
    let mut comp = vec![0i64; ncols];
    let mut uncomp = vec![0i64; ncols];
    let mut encs: Vec<std::collections::BTreeSet<String>> =
        (0..ncols).map(|_| std::collections::BTreeSet::new()).collect();

    for rg in meta.row_groups() {
        for (c, col) in rg.columns().iter().enumerate() {
            comp[c] += col.compressed_size();
            uncomp[c] += col.uncompressed_size();
            for e in col.encodings() {
                encs[c].insert(format!("{:?}", e));
            }
        }
    }
    let tot_c: i64 = comp.iter().sum();
    let tot_u: i64 = uncomp.iter().sum();

    println!(
        "{:30} {:>11} {:>12} {:>6}  encodings",
        "column", "comp(MB)", "uncomp(MB)", "ratio"
    );
    for c in 0..ncols {
        println!(
            "{:30} {:>11.2} {:>12.2} {:>6.2}  {}",
            names[c],
            comp[c] as f64 / 1e6,
            uncomp[c] as f64 / 1e6,
            uncomp[c] as f64 / (comp[c].max(1) as f64),
            encs[c].iter().cloned().collect::<Vec<_>>().join(",")
        );
    }
    println!(
        "{:30} {:>11.2} {:>12.2}",
        "TOTAL",
        tot_c as f64 / 1e6,
        tot_u as f64 / 1e6
    );
    println!("rows={} row_groups={}", meta.file_metadata().num_rows(), nrg);
    Ok(())
}

pub fn arrow_main(zpath: &str, opath: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut e = EchtVars::open(zpath);

    let chunks = discover_chunks(&mut e)?;
    eprintln!("[echtvar arrow] {} chunks to convert", chunks.len());

    let mut writer = ArrowWriterEchtvar::create(opath, &e.fields, &e.strings)?;

    // synthetic rid per contig so set_position's cache always reloads.
    let mut synthetic_rid: i32 = i32::MIN;
    let mut last_chrom: Option<String> = None;
    let mut n_chunks = 0u64;
    let mut n_vars = 0u64;
    let mut n_long = 0u64;

    for (chrom, chunk_id) in &chunks {
        if last_chrom.as_deref() != Some(chrom.as_str()) {
            synthetic_rid = synthetic_rid.saturating_add(1);
            last_chrom = Some(chrom.clone());
        }
        let chunk_base: u32 = chunk_id << 20;
        e.set_position(synthetic_rid, chrom.clone(), chunk_base)?;

        if e.var32s.is_empty() {
            continue;
        }

        // Build the long-variant ordinal map. In the zip reader, each
        // LongVariant.idx is the ordinal into the (sorted) var32s array, and
        // the sequence is kmer16-packed ref/alt. We decode it back to raw bytes
        // here so the parquet format stores plain ref/alt.
        let mut longs: HashMap<usize, (u32, Vec<u8>, Vec<u8>)> = HashMap::new();
        for l in &e.longs {
            let (r, a) = kmer16::decode_var(&l.sequence);
            longs.insert(l.idx as usize, (l.position, r, a));
        }
        n_long += longs.len() as u64;

        writer.write_chunk(chrom, *chunk_id, &e.var32s, &e.values, &longs)?;
        n_chunks += 1;
        n_vars += e.var32s.len() as u64;
        if n_chunks % 500 == 0 {
            eprintln!("[echtvar arrow] {} chunks, {} variants written", n_chunks, n_vars);
        }
    }

    writer.close()?;
    eprintln!(
        "[echtvar arrow] done: {} chunks, {} variants ({} long) -> {}",
        n_chunks, n_vars, n_long, opath
    );
    Ok(())
}
