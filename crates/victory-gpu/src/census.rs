//! Diagnostics: how often each problem logged once (`once!`) happens. The
//! warning says that it happens; this says how often, per frame, at its
//! file and line. Nothing here changes what the layer does.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Each site's counter, file and line, registered the first time it fires.
static SITES: Mutex<Vec<(&'static AtomicU64, &'static str, u32)>> = Mutex::new(Vec::new());

pub(crate) fn register(count: &'static AtomicU64, file: &'static str, line: u32) {
    SITES.lock().unwrap_or_else(|e| e.into_inner()).push((count, file, line));
}

/// Logs the sites that fired since the last line, most often first, and
/// zeroes their counts.
pub(crate) fn line(frames: f64) {
    let sites = SITES.lock().unwrap_or_else(|e| e.into_inner());
    let mut fired: Vec<(u64, &str, u32)> = sites
        .iter()
        .map(|(c, file, line)| (c.swap(0, Relaxed), *file, *line))
        .filter(|f| f.0 > 0)
        .collect();
    drop(sites);
    if fired.is_empty() {
        return;
    }
    fired.sort_by_key(|f| std::cmp::Reverse(f.0));
    let text: Vec<String> = fired
        .iter()
        .map(|(n, file, line)| format!("{file}:{line} {:.1}/frame", *n as f64 / frames.max(1.0)))
        .collect();
    tracing::info!("problems logged once, how often since the last line: {}", text.join(", "));
}
