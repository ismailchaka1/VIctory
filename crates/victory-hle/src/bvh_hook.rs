//! `VICTORY_BVH_LOG=1`: the title's calls into its ray-tracing structure
//! builder (`XboxBvhBuilder.dll`) logged with their arguments, then passed
//! on unchanged. For finding where the builder's results go and what the
//! title reads back (the builder's own layout of `BATCHED_DESC2` is not
//! public: its qwords are logged raw, and what each that looks like a
//! pointer points at).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering::Relaxed};

type This = *mut c_void;

/// The real exports, by the thunks' order.
static REAL: [AtomicUsize; 5] = [const { AtomicUsize::new(0) }; 5];

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_LOG").is_some_and(|v| v == "1"))
}

/// The thunk standing in for the export `name` (at `address`), if it is one
/// that is logged.
pub fn wrap(name: &str, address: usize) -> Option<usize> {
    let (k, thunk) = if name.starts_with("?BuildRaytracingAccelerationStructureBatched2X@") {
        (0, build as *const () as usize)
    } else if name.starts_with("?CopyRaytracingAccelerationStructure@") {
        (1, copy as *const () as usize)
    } else if name.starts_with("?GetRaytracingAccelerationStructurePrebuildInfo@") {
        (2, prebuild as *const () as usize)
    } else if name.starts_with("?RehydrateRaytracingAccelerationStructureBatched2X@") {
        (3, rehydrate as *const () as usize)
    } else if name.starts_with("?Set64BitMortonCodes@") {
        (4, morton as *const () as usize)
    } else {
        return None;
    };
    REAL[k].store(address, Relaxed);
    tracing::info!(name, real = format_args!("{address:#x}"), "BVH builder export logged");
    Some(thunk)
}

/// At most this many lines in all.
fn more() -> bool {
    static LINES: AtomicU32 = AtomicU32::new(0);
    LINES.fetch_add(1, Relaxed) < 3000
}

/// `n` qwords at `at` (title memory; null gives none).
fn qwords(at: *const u64, n: usize) -> Vec<u64> {
    if at.is_null() {
        return Vec::new();
    }
    // SAFETY: the title passes readable structures of at least this size
    // (the callers ask only for what the arguments describe, or for one
    // level of what looks like a pointer into memory the title uses).
    (0..n).map(|i| unsafe { at.add(i).read_unaligned() }).collect()
}

fn looks_like_pointer(v: u64) -> bool {
    (0x1_0000..1 << 47).contains(&v) && v % 8 == 0 && readable(v as usize)
}

/// Whether 64 bytes at `at` lie in committed readable memory of one page.
fn readable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    if (at & 0xfff) > 0x1000 - 64 {
        return false;
    }
    // SAFETY: a zeroed MEMORY_BASIC_INFORMATION is valid; this reads the
    // address space's bookkeeping only.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect != 0 && info.Protect & (PAGE_GUARD | PAGE_NOACCESS) == 0
}

fn hex(words: &[u64]) -> String {
    words.iter().map(|w| format!("{w:#x}")).collect::<Vec<_>>().join(" ")
}

/// The 0x100 bytes before each structure the title names (where the
/// title keeps its own head of a structure: structures start 0x100 into
/// their page), taken at the call, compared by a thread of ours until
/// they change: to find out whether the builder's results land there.
struct Head {
    at: usize,
    kind: &'static str,
    words: [u64; 32],
    done: bool,
}

static HEADS: std::sync::Mutex<Vec<Head>> = std::sync::Mutex::new(Vec::new());

/// The first rehydrated structures in full (head and first 0x400 bytes),
/// at the call and when their head changes.
struct Dump {
    at: usize,
    words: Vec<u64>,
    done: bool,
}

static DUMPS: std::sync::Mutex<Vec<Dump>> = std::sync::Mutex::new(Vec::new());
const DUMP_WORDS: usize = 0x500 / 8;

fn dump_words(at: usize) -> Option<Vec<u64>> {
    (0..DUMP_WORDS * 8 / 64).all(|k| readable(at + k * 64)).then(|| qwords(at as *const u64, DUMP_WORDS))
}

fn log_dump(dest: usize, when: &str, words: &[u64]) {
    for (k, chunk) in words.chunks(16).enumerate() {
        tracing::info!(
            dest = format_args!("{dest:#x}"),
            when,
            from = format_args!("{:+#x}", k as isize * 128 - 0x100),
            words = hex(chunk),
            "BVH structure dump"
        );
    }
}

fn track_dump(dest: u64) {
    let Some(at) = (dest as usize).checked_sub(0x100) else { return };
    let mut dumps = DUMPS.lock().unwrap_or_else(|e| e.into_inner());
    if dumps.len() >= 4 || dumps.iter().any(|d| d.at == at) {
        return;
    }
    let Some(words) = dump_words(at) else { return };
    log_dump(at + 0x100, "at the call", &words);
    dumps.push(Dump { at, words, done: false });
}

fn check_dumps() {
    static CHANGES: AtomicU32 = AtomicU32::new(0);
    let mut dumps = DUMPS.lock().unwrap_or_else(|e| e.into_inner());
    for d in dumps.iter_mut().filter(|d| !d.done) {
        let Some(now) = dump_words(d.at) else { continue };
        if now != d.words {
            // Each change, up to 24 in all: when the streamed data arrives
            // and when the builder's nodes do.
            let head = now[..32] != d.words[..32];
            let body_bytes = now[32..].iter().filter(|&&w| w != 0).count() * 8;
            tracing::info!(dest = format_args!("{:#x}", d.at + 0x100), head_changed = head, body_nonzero_bytes = body_bytes, "BVH structure changed");
            if CHANGES.fetch_add(1, Relaxed) < 24 {
                log_dump(d.at + 0x100, if head { "after its head changed" } else { "after its body changed" }, &now);
            }
            d.words = now;
            if head {
                d.done = true;
            }
        }
    }
}

/// Post-build info the builder's GPU stages write and the title's CPU then
/// reads (compacted sizes): where, at the call, with the build's
/// destination and the result size its prebuild promised (0: no prebuild
/// seen for its inputs), and the value there at the call.
struct Post {
    at: usize,
    kind: u32,
    dest: u64,
    max: u64,
    was: u64,
    /// When the call was; 2 s later (seen from a later build call) the
    /// value is logged whether or not it changed.
    when: std::time::Instant,
    told: bool,
    /// A later build named the same address before this one was told: the
    /// value then may be that build's.
    reused: bool,
    /// Counted as landed by the watcher (kept for the compaction check).
    counted: bool,
}

static POSTS: std::sync::Mutex<Vec<Post>> = std::sync::Mutex::new(Vec::new());
/// Prebuild results by the inputs' descs address.
static PREBUILT: std::sync::Mutex<Option<std::collections::HashMap<u64, u64>>> = std::sync::Mutex::new(None);
/// Post-build values landed: in all, 0, past the prebuild's result size.
static LANDED: [AtomicU32; 3] = [const { AtomicU32::new(0) }; 3];

/// Post-build entries more than 2 s old, each logged once: their value now,
/// and the first qwords of their structure (on the build thread: the
/// watcher thread's ticks are too slow to count on, run track5-5).
fn tell_old_posts() {
    static SAID: AtomicU32 = AtomicU32::new(0);
    let mut posts = POSTS.lock().unwrap_or_else(|e| e.into_inner());
    for p in posts.iter_mut().filter(|p| !p.told && p.when.elapsed() > std::time::Duration::from_secs(2)) {
        p.told = true;
        if SAID.fetch_add(1, Relaxed) >= 3000 {
            continue;
        }
        let readable_now = readable(p.at);
        let now = if readable_now { qwords(p.at as *const u64, 1)[0] } else { 0 };
        let head = if readable(p.dest as usize) { hex(&qwords(p.dest as *const u64, 4)) } else { "unreadable".into() };
        tracing::warn!(
            at = format_args!("{:#x}", p.at),
            readable = readable_now,
            reused = p.reused,
            was = format_args!("{:#x}", p.was),
            now = format_args!("{now:#x}"),
            result_max = format_args!("{:#x}", p.max),
            dest = format_args!("{:#x}", p.dest),
            dest_head = head,
            "BVH post-build info 2 s after the call"
        );
    }
}

fn track_posts(w: &[u64]) {
    // w[1..4]: the inputs (type|flags, count|layout, descs); w[6] post-build count, w[7] their descs.
    let (n, descs) = (w[6] as usize, w[7]);
    if n == 0 || n > 16 || !looks_like_pointer(descs) {
        return;
    }
    let max = PREBUILT.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&w[3]).copied()).unwrap_or(0);
    let d = qwords(descs as *const u64, 2 * n);
    let mut posts = POSTS.lock().unwrap_or_else(|e| e.into_inner());
    for k in 0..n {
        let (at, kind) = (d[2 * k] as usize, d[2 * k + 1] as u32);
        if posts.len() < 20_000 && at % 8 == 0 && readable(at) {
            let was = qwords(at as *const u64, 1)[0];
            for old in posts.iter_mut().filter(|o| o.at == at && !o.told) {
                old.reused = true;
            }
            posts.push(Post { at, kind, dest: w[0], max, was, when: std::time::Instant::now(), told: false, reused: false, counted: false });
        }
    }
}

fn check_posts() {
    static SAID: AtomicU32 = AtomicU32::new(0);
    let mut posts = POSTS.lock().unwrap_or_else(|e| e.into_inner());
    posts.retain_mut(|p| {
        if p.counted {
            return true;
        }
        let readable_now = readable(p.at);
        if !readable_now {
            return true;
        }
        let now = qwords(p.at as *const u64, 1)[0];
        if now == p.was {
            return true;
        }
        p.counted = true;
        LANDED[0].fetch_add(1, Relaxed);
        let zero = now == 0;
        let past = p.max > 0 && now > p.max;
        if zero {
            LANDED[1].fetch_add(1, Relaxed);
        }
        if past {
            LANDED[2].fetch_add(1, Relaxed);
        }
        if (zero || past || SAID.load(Relaxed) < 20) && SAID.fetch_add(1, Relaxed) < 2000 {
            tracing::warn!(
                at = format_args!("{:#x}", p.at),
                kind = p.kind,
                dest = format_args!("{:#x}", p.dest),
                value = format_args!("{now:#x}"),
                result_max = format_args!("{:#x}", p.max),
                zero,
                past,
                "BVH post-build info landed"
            );
        }
        true
    });
}

fn head_words(at: usize) -> Option<[u64; 32]> {
    if (0..4).all(|k| readable(at + k * 64)) {
        let w = qwords(at as *const u64, 32);
        Some(std::array::from_fn(|i| w[i]))
    } else {
        None
    }
}

fn track_head(dest: u64, kind: &'static str) {
    let Some(at) = (dest as usize).checked_sub(0x100) else { return };
    let Some(words) = head_words(at) else { return };
    let mut heads = HEADS.lock().unwrap_or_else(|e| e.into_inner());
    if heads.len() >= 4000 || heads.iter().any(|h| h.at == at) {
        return;
    }
    if heads.len() < 5 {
        tracing::info!(dest = format_args!("{dest:#x}"), kind, head = hex(&words), "BVH structure head at the call");
    }
    heads.push(Head { at, kind, words, done: false });
    static WATCHER: std::sync::Once = std::sync::Once::new();
    WATCHER.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-bvh-heads".into()).spawn(|| {
            let mut ticks = 0u32;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                ticks += 1;
                check_dumps();
                check_posts();
                // The scan runs without the lock: the title's builder calls take it (track_head),
                // and a scan of thousands of heads held it for most of each tick (run-vis-52: the
                // title's build thread waited on it until the title's hang detector stopped it).
                let pending: Vec<(usize, &'static str, [u64; 32])> = {
                    let heads = HEADS.lock().unwrap_or_else(|e| e.into_inner());
                    heads.iter().filter(|h| !h.done).map(|h| (h.at, h.kind, h.words)).collect()
                };
                let mut found = Vec::new();
                for (at, kind, words) in pending {
                    let Some(now) = head_words(at) else { continue };
                    if now != words {
                        found.push((at, kind, words, now));
                    }
                }
                let mut heads = HEADS.lock().unwrap_or_else(|e| e.into_inner());
                for h in heads.iter_mut() {
                    if found.iter().any(|f| f.0 == h.at) {
                        h.done = true;
                    }
                }
                let changed = heads.iter().filter(|h| h.done).count();
                let tracked = heads.len();
                drop(heads);
                let mut boxes = 0;
                for (at, kind, words, now) in found {
                    let h = Head { at, kind, words, done: true };
                    let first = (0..32).find(|&i| now[i] != h.words[i]).unwrap_or(0);
                    let boxy = now.iter().any(|&w| {
                        let lo = w as u32;
                        let hi = (w >> 32) as u32;
                        [lo, hi].iter().any(|&d| d == 0x7f80_0000 || d == 0xff80_0000)
                            || (0..4).any(|k| matches!((w >> (16 * k)) as u16, 0x7c00 | 0xfc00))
                    });
                    if boxy {
                        boxes += 1;
                    }
                    tracing::warn!(
                        dest = format_args!("{:#x}", h.at + 0x100),
                        kind = h.kind,
                        first_changed = format_args!("{:#x}", first * 8),
                        was = format_args!("{:#x}", h.words[first]),
                        now = format_args!("{:#x}", now[first]),
                        boxy,
                        head = hex(&now[..8]),
                        "BVH structure head changed after the call"
                    );
                }
                if ticks % 50 == 0 {
                    tracing::info!(tracked, changed, boxes, "BVH structure heads");
                    let [all, zero, past] = LANDED.each_ref().map(|a| a.load(Relaxed));
                    let waiting = POSTS.lock().unwrap_or_else(|e| e.into_inner()).len();
                    tracing::info!(all, zero, past, waiting, "BVH post-build info landed so far");
                }
            }
        });
    });
}

extern "system" fn build(this: This, list: This, count: u32, descs: *const u64) {
    for i in 0..count as usize {
        // SAFETY: the title passes `count` descs of 10 qwords (run vis-33).
        let w = qwords(unsafe { descs.add(i * 10) }, 10);
        track_head(w[0], "build");
        track_posts(&w);
        tell_old_posts();
    }
    if more() {
        let raw = qwords(descs, 24);
        tracing::info!(count, descs = format_args!("{descs:p}"), raw = hex(&raw), "BVH build (batched 2)");
        for (i, &w) in raw.iter().enumerate() {
            if looks_like_pointer(w) && more() {
                tracing::info!(word = i, at = format_args!("{w:#x}"), points_at = hex(&qwords(w as *const u64, 8)), "BVH build: a pointer in the desc");
            }
        }
    }
    // SAFETY: the export this thunk stands for, with the same arguments.
    let real: extern "system" fn(This, This, u32, *const u64) =
        unsafe { std::mem::transmute(REAL[0].load(Relaxed)) };
    real(this, list, count, descs);
}

extern "system" fn copy(this: This, list: This, dest: u64, source: u64, mode: u32) {
    track_head(dest, "copy");
    // At a compaction (mode 1) the title has waited for the build: the size
    // it compacts with is the post-build value, which must have landed.
    if mode == 1 {
        static SAID: AtomicU32 = AtomicU32::new(0);
        static KINDS: [AtomicU32; 4] = [const { AtomicU32::new(0) }; 4];
        let posts = POSTS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = posts.iter().rev().find(|p| p.dest == source) {
            let now = if readable(p.at) { qwords(p.at as *const u64, 1)[0] } else { 0 };
            let landed = now != p.was;
            let past = p.max > 0 && now > p.max;
            let k = match (landed, now == 0, past) {
                (true, _, false) => 0,
                (true, _, true) => 1,
                (false, true, _) => 2,
                (false, false, _) => 3,
            };
            let n = KINDS[k].fetch_add(1, Relaxed);
            if (k != 0 || n < 5) && SAID.fetch_add(1, Relaxed) < 1000 {
                tracing::warn!(
                    source = format_args!("{source:#x}"),
                    size = format_args!("{now:#x}"),
                    was = format_args!("{:#x}", p.was),
                    result_max = format_args!("{:#x}", p.max),
                    landed,
                    past,
                    totals = ?KINDS.each_ref().map(|a| a.load(Relaxed)),
                    "BVH compaction: the post-build size at the copy (totals: landed ok, landed past max, not landed 0, not landed stale)"
                );
            }
        }
    }
    if more() {
        tracing::info!(dest = format_args!("{dest:#x}"), source = format_args!("{source:#x}"), mode, "BVH copy");
    }
    // SAFETY: as above.
    let real: extern "system" fn(This, This, u64, u64, u32) = unsafe { std::mem::transmute(REAL[1].load(Relaxed)) };
    real(this, list, dest, source, mode);
}

/// `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS`: type, flags,
/// count, layout, then the descs' address.
fn inputs(at: *const u64) -> String {
    let w = qwords(at, 3);
    if w.len() < 3 {
        return "none".into();
    }
    format!(
        "type {} flags {:#x} descs {} layout {} at {:#x}",
        w[0] as u32,
        w[0] >> 32,
        w[1] as u32,
        w[1] >> 32,
        w[2]
    )
}

extern "system" fn prebuild(this: This, input: *const u64, stream: *const u64, out: *mut u64) {
    // SAFETY: as above.
    let real: extern "system" fn(This, *const u64, *const u64, *mut u64) =
        unsafe { std::mem::transmute(REAL[2].load(Relaxed)) };
    real(this, input, stream, out);
    if let (Some(&descs), Some(&max)) = (qwords(input, 3).get(2), qwords(out, 1).first()) {
        PREBUILT.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(descs, max);
    }
    if more() {
        let sizes = qwords(out, 3);
        tracing::info!(
            inputs = inputs(input),
            stream = hex(&qwords(stream, 4)),
            result_max = sizes.first().copied().unwrap_or(0),
            scratch = sizes.get(1).copied().unwrap_or(0),
            update_scratch = sizes.get(2).copied().unwrap_or(0),
            "BVH prebuild info"
        );
    }
}

extern "system" fn rehydrate(this: This, list: This, count: u32, descs: *const u64, sizes: *const u64) -> i32 {
    for i in 0..count as usize {
        // SAFETY: the title passes `count` descs of 6 qwords.
        let dest = unsafe { descs.add(i * 6).read_unaligned() };
        track_dump(dest);
        track_head(dest, "rehydrate");
    }
    if more() {
        tracing::info!(count, sizes = hex(&qwords(sizes, (count as usize).min(8))), "BVH rehydrate (batched 2)");
        for i in 0..(count as usize).min(8) {
            // `D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC`: 6 qwords.
            // SAFETY: the title passes `count` descs.
            let d = qwords(unsafe { descs.add(i * 6) }, 6);
            // SAFETY: the title passes `count` of these too.
            let other = if sizes.is_null() { 0 } else { unsafe { sizes.add(i).read_unaligned() } };
            if more() {
                tracing::info!(
                    i,
                    dest = format_args!("{:#x}", d[0]),
                    inputs = inputs(unsafe { descs.add(i * 6 + 1) }),
                    source = format_args!("{:#x}", d[4]),
                    scratch = format_args!("{:#x}", d[5]),
                    second = format_args!("{:#x}", other),
                    // the serialized data the builder sizes its dispatches from, as the call finds it
                    src_head = if readable(d[4] as usize) { hex(&qwords(d[4] as *const u64, 8)) } else { "unreadable".into() },
                    dest_page = ?crate::modules::xmem::physical_page(d[0] as usize),
                    second_page = ?crate::modules::xmem::physical_page(other as usize),
                    "BVH rehydrate desc"
                );
            }
        }
    }
    // SAFETY: as above.
    let real: extern "system" fn(This, This, u32, *const u64, *const u64) -> i32 =
        unsafe { std::mem::transmute(REAL[3].load(Relaxed)) };
    let hr = real(this, list, count, descs, sizes);
    if more() {
        tracing::info!(hr = format_args!("{hr:#x}"), "BVH rehydrate returned");
    }
    hr
}

extern "system" fn morton(this: This, on: u8) {
    tracing::info!(on, "BVH builder: 64-bit Morton codes");
    // SAFETY: as above.
    let real: extern "system" fn(This, u8) = unsafe { std::mem::transmute(REAL[4].load(Relaxed)) };
    real(this, on);
}
