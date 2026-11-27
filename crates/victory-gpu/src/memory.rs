//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES, MEMORY_TOUCHED, MEMORY_WRITTEN, SLOTS_READ};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows; the
/// structure builder ran out of 1,023 within 2 seconds (run vis-5), and a
/// page left without a number reads zeros.
const COPIES: u32 = 8191;

/// Frames between looks for copies no shader touched ([`Memory::retire`]).
const RETIRE_EVERY: u64 = 32;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    /// `None`: a number given up ([`Memory::retire`]), free for the next.
    copies: Vec<Option<Arc<Shadow>>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Why the next copies are numbered (diagnostics: logged with each).
    why: &'static str,
    /// Directory pages shaders ever stored to ([`MEMORY_WRITTEN`]), and per
    /// copy the runs of them it holds (title addresses), made again each
    /// frame, and whether they held still since the frame before (else
    /// the copy is declared whole).
    written: std::collections::BTreeSet<u64>,
    runs: Vec<(Vec<(u64, u64)>, bool)>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS + MEMORY_WRITTEN + MEMORY_TOUCHED), "faults")?,
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            why: "",
            written: Default::default(),
            runs: Vec::new(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        if unsafe { entry.read() } != 0 {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| c.as_ref().is_some_and(|c| Arc::ptr_eq(c, &shadow))) {
            Some(n) => n,
            None if self.copies.iter().any(Option::is_none) || self.copies.len() < COPIES as usize => {
                let n = match self.copies.iter().position(Option::is_none) {
                    Some(n) => {
                        self.copies[n] = Some(shadow.clone());
                        // Its view is made again in every heap.
                        for made in self.heaps.values_mut() {
                            made.1 = made.1.min(n as u32);
                        }
                        n
                    }
                    None => {
                        self.copies.push(Some(shadow.clone()));
                        self.copies.len() - 1
                    }
                };
                let (first, len) = shadow.span();
                tracing::info!(
                    n,
                    address = format_args!("{address:#x}"),
                    first = format_args!("{first:#x}"),
                    len = format_args!("{len:#x}"),
                    why = self.why,
                    "title memory by address: a copy numbered"
                );
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    let mb = self.copies.iter().flatten().map(|c| c.span().1).sum::<u64>() >> 20;
                    tracing::warn!(COPIES, mb, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
    }

    /// Gives up the copies no shader touched since the last look
    /// ([`MEMORY_TOUCHED`]): their pages are emptied and their numbers
    /// freed. Every numbered copy was held for the whole run, and with the
    /// builder on that was ~2,900 windows of 4 MiB (11.5 GB) in two minutes
    /// and the run died out of memory (runs 79, 81, 82). A page touched again
    /// faults and is mapped anew next frame (those accesses read 0).
    fn retire(&mut self) {
        let touched = |page: u64| (MEMORY_FAULTS + MEMORY_WRITTEN) as usize + page as usize;
        let mut freed = (0usize, 0u64);
        for n in 0..self.copies.len() {
            let Some(shadow) = self.copies[n].as_ref() else {
                continue;
            };
            let (first, len) = shadow.span();
            let pages = first / PAGE..(first + len).div_ceil(PAGE).min(u64::from(MEMORY_PAGES) - 1);
            let mut used = false;
            for page in pages.clone() {
                // SAFETY: inside the faults' mapping (pages below MEMORY_PAGES);
                // a mark the GPU makes meanwhile is a frame's access lost.
                let at = unsafe { self.faults.1.add(touched(page)) };
                // SAFETY: as above.
                unsafe {
                    if at.read_volatile() != 0 {
                        used = true;
                        at.write_volatile(0);
                    }
                }
            }
            if used {
                continue;
            }
            for page in pages {
                // SAFETY: inside the directory's mapping.
                let entry = unsafe { self.directory.1.add(page as usize * 2) };
                // SAFETY: as above.
                unsafe {
                    if entry.read_volatile() == n as u32 + 1 {
                        entry.write_volatile(0);
                    }
                }
            }
            freed = (freed.0 + 1, freed.1 + len);
            self.copies[n] = None;
        }
        if freed.0 > 0 {
            tracing::info!(copies = freed.0, mb = freed.1 >> 20, "title memory by address: copies no shader touched are given up");
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    copies = self.copies.len(),
                    mb = self.copies.iter().flatten().map(|c| c.span().1).sum::<u64>() >> 20,
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        self.why = "a shader followed a pointer there";
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// The pages shaders stored to since the last look join [`Memory::written`].
    fn take_written(&mut self) {
        for w in 0..MEMORY_WRITTEN as usize {
            // SAFETY: inside the faults' mapping, past the fault numbers; a
            // bit the GPU sets meanwhile is seen next time.
            let at = unsafe { self.faults.1.add(MEMORY_FAULTS as usize + w) };
            // SAFETY: as above.
            let bits = unsafe { at.read_volatile() };
            if bits == 0 {
                continue;
            }
            for b in 0..32 {
                if bits & (1 << b) != 0 {
                    self.written.insert((w as u64 * 32 + b) * PAGE);
                }
            }
        }
        let old = std::mem::take(&mut self.runs);
        self.runs = self
            .copies
            .iter()
            .enumerate()
            .map(|(n, c)| {
                let now = c.as_ref().map_or_else(Vec::new, |c| runs(&self.written, c.span()));
                let still = old.get(n).is_some_and(|o| o.0 == now);
                (now, still)
            })
            .collect();
        #[cfg(feature = "diagnostics")]
        {
            let mb = |f: &dyn Fn(&(Vec<(u64, u64)>, bool), &Arc<Shadow>) -> u64| {
                self.runs.iter().zip(&self.copies).filter_map(|(r, c)| c.as_ref().map(|c| f(r, c))).sum::<u64>() >> 20
            };
            tracing::info!(
                copies = self.copies.len(),
                whole = self.runs.iter().filter(|r| !r.1).count(),
                whole_mb = mb(&|r, c| if r.1 { 0 } else { c.span().1 }),
                narrow_mb = mb(&|r, _| if r.1 { r.0.iter().map(|(a, b)| b - a).sum() } else { 0 }),
                pages = self.written.len(),
                "title memory by address: copies declared written whole (their written pages still growing) and by page"
            );
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.why = "a dispatch named it";
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(started);
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        if frame % RETIRE_EVERY == 0 {
            m.retire();
        }
        m.take_faults();
        m.take_written();
    }
    let numbered = m.copies.len();
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let Some(shadow) = shadow else {
            continue;
        };
        let (_, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    for (n, shadow) in m.copies.iter().enumerate() {
        let Some(shadow) = shadow else {
            continue;
        };
        let (start, len) = shadow.span();
        if first_read || n >= numbered {
            shadow.refresh(list, start, len);
        }
        // Only the pages shaders were seen to store to, once the copy's set
        // of them held still for a frame; whole before that (new copies, and
        // copies with pages stored to for the first time).
        // shortcut: the first store to a new page of a copy whose set held
        // still is not copied back; a page-exact put-back would close that.
        if writes && (first_write || n >= numbered) {
            match m.runs.get(n) {
                Some((runs, true)) if by_page() => {
                    for &(from, to) in runs {
                        shadow.gpu_writes(list, from, to - from);
                    }
                }
                _ => shadow.gpu_writes(list, start, len),
            }
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

/// The [`SLOTS_READ`] bitmap: one bit per view-heap slot shaders picked a
/// texture from. It lives in the card's memory (shaders read it per pixel;
/// in the CPU-mapped faults buffer that cost run t1-10 76 ms a frame) and
/// once a frame [`take_slots_read`] copies it out and clears it.
struct SlotsRead {
    bits: ID3D12Resource,
    zeros: ID3D12Resource,
    back: ID3D12Resource,
    /// The copy on its way out: its fence value.
    out: Option<u64>,
}

// SAFETY: D3D12 resources may be used from any thread.
unsafe impl Send for SlotsRead {}

static SLOTS: Mutex<Option<Option<SlotsRead>>> = Mutex::new(None);

/// The bitmap's GPU address, for a shader that notes the heap slots its
/// index-picked textures name; `list` holds it.
pub(crate) fn slots_read(list: &HostList) -> Option<u64> {
    let mut guard = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    let s = guard
        .get_or_insert_with(|| {
            let host = crate::host::host()?;
            let bytes = u64::from(SLOTS_READ) * 4;
            Some(SlotsRead { bits: host.uav_buffer(bytes)?, zeros: host.uav_buffer(bytes)?, back: host.read_back_buffer(bytes)?, out: None })
        })
        .as_ref()?;
    const KEPT: usize = usize::MAX - 0x6f;
    if list.first_use(KEPT) {
        list.keep(&s.bits);
    }
    // SAFETY: a live buffer.
    Some(unsafe { s.bits.GetGPUVirtualAddress() })
}

/// The [`SLOTS_READ`] bitmap of the last copy that landed (bit `i % 32` of
/// word `i / 32`: a shader picked a texture from slot `i`), and sends the
/// next copy, which also clears it. None before any shader had the buffer,
/// or while the last copy is still on the card.
/// shortcut: bits set between a copy and its clear are lost, and the bits
/// come a frame or two late; the slot is read again the next frame it is
/// on screen.
pub(crate) fn take_slots_read() -> Option<Vec<u32>> {
    let host = crate::host::host()?;
    let mut guard = SLOTS.lock().unwrap_or_else(|e| e.into_inner());
    let s = guard.as_mut()?.as_mut()?;
    let bytes = u64::from(SLOTS_READ) * 4;
    let mut bits = None;
    if let Some(value) = s.out {
        if !host.landed(value) {
            return None;
        }
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer the card is done with; mapped for the
        // read and unmapped with nothing written.
        unsafe {
            if s.back.Map(0, Some(&D3D12_RANGE { Begin: 0, End: bytes as usize }), Some(&mut data)).is_ok() {
                bits = Some(std::slice::from_raw_parts(data.cast::<u32>(), SLOTS_READ as usize).to_vec());
                s.back.Unmap(0, Some(&D3D12_RANGE::default()));
            }
        }
        s.out = None;
    }
    s.out = host.copy_out_and_clear(&s.bits, &s.back, &s.zeros, bytes);
    bits
}

/// After a submission landed: pages shaders stored to through raw addresses
/// that no copy had declared written (new pages of a copy declared by page)
/// get the copy's bytes put back whole, and are declared from now on.
/// Skipped if the memory is busy; the next landing looks again.
/// shortcut: a title write to such a page between its refresh and now is
/// overwritten by the copy's bytes; the builder's pages are the GPU's alone.
pub(crate) fn land_new_writes() {
    if !by_page() {
        return;
    }
    // Tried for up to 50 ms, not once: skipped, the pages landed after the
    // fence the title waits on was set (the structure builder's post-build
    // sizes, read by the title's compaction, arrived seconds late in run
    // track5-7). Not blocking: a recording thread holding the memory while
    // it waits for a landing would never let go.
    let began = std::time::Instant::now();
    let mut guard = loop {
        match MEMORY.try_lock() {
            Ok(g) => break g,
            Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) if began.elapsed() < std::time::Duration::from_millis(50) => std::thread::yield_now(),
            Err(std::sync::TryLockError::WouldBlock) => {
                use std::sync::atomic::{AtomicU64, Ordering};
                static SKIPPED: AtomicU64 = AtomicU64::new(0);
                let n = SKIPPED.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(n, "title memory by address: new shader writes not landed (the memory was busy for 50 ms); the next landing takes them");
                }
                return;
            }
        }
    };
    let Some(m) = guard.as_mut() else {
        return;
    };
    for w in 0..MEMORY_WRITTEN as usize {
        // SAFETY: inside the faults' mapping, past the fault numbers.
        let bits = unsafe { m.faults.1.add(MEMORY_FAULTS as usize + w).read_volatile() };
        if bits == 0 {
            continue;
        }
        for b in 0..32u64 {
            let page = w as u64 * 32 + b;
            if bits & (1 << b) == 0 || !m.written.insert(page * PAGE) {
                continue;
            }
            // SAFETY: inside the directory's mapping (pages below MEMORY_PAGES).
            let copy = unsafe { m.directory.1.add(page as usize * 2).read_volatile() };
            if let Some(shadow) = (copy as usize).checked_sub(1).and_then(|n| m.copies.get(n)?.as_ref()) {
                shadow.put_back((page * PAGE) as usize, PAGE as usize);
            }
        }
    }
}

/// The runs of `pages` (64 KiB each, by title address) within `[start,
/// start + len)`, cut to it.
fn runs(pages: &std::collections::BTreeSet<u64>, (start, len): (u64, u64)) -> Vec<(u64, u64)> {
    let end = start + len;
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &p in pages.range(start / PAGE * PAGE..end) {
        let (from, to) = (p.max(start), (p + PAGE).min(end));
        match out.last_mut() {
            Some(last) if last.1 >= from => last.1 = to,
            _ => out.push((from, to)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn runs_join_and_cut() {
        let pages: std::collections::BTreeSet<u64> = [0x10000, 0x20000, 0x50000].into_iter().collect();
        assert_eq!(super::runs(&pages, (0x18000, 0x40000)), vec![(0x18000, 0x30000), (0x50000, 0x58000)]);
        assert_eq!(super::runs(&pages, (0, 0x100000)), vec![(0x10000, 0x30000), (0x50000, 0x60000)]);
    }
}

/// Diagnostics: `ready`'s time, counted as list-call time of pseudo-line
/// 1000000 (`copystat::recorded_at`), whichever way it returns.
#[cfg(feature = "diagnostics")]
struct Timed(std::time::Instant);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        crate::copystat::by_caller(self.0, 1_000_000);
    }
}

/// Whether numbered copies are declared written by page
/// ([`MEMORY_WRITTEN`]); `VICTORY_MEMORY_PAGES=0` declares each whole, as
/// before (each dispatch that can store: every copy, every byte compared).
fn by_page() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_PAGES").is_none_or(|v| v != "0"))
}


