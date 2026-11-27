//! Query heaps. Each has a host query heap of the same type and size,
//! made on its first use by a list (`host_heap`); the list methods record
//! the queries on the host list and resolve them there (`list.rs`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap>>,
}

/// The host heap of the title's heap `this`, made on first use; `None`
/// without a host, or for a heap type the host does not make.
#[cfg(windows)]
pub(crate) fn host_heap<'a>(this: This) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap> {
    use windows::Win32::Graphics::Direct3D12::*;
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our heaps.
    let heap = unsafe { &*this.cast::<Heap>() };
    heap.host
        .get_or_init(|| {
            let host = crate::host::host()?;
            let desc = D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE(heap.desc.kind as i32),
                Count: heap.desc.count,
                NodeMask: 0,
            };
            let mut made: Option<ID3D12QueryHeap> = None;
            // SAFETY: a valid description and out pointer.
            match unsafe { host.device.CreateQueryHeap(&desc, &mut made) } {
                Ok(()) => made,
                Err(e) => {
                    tracing::warn!(?heap.desc, error = %e, "host query heap refused; its queries get plausible values");
                    None
                }
            }
        })
        .as_ref()
}

/// Device method.
pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

impl Drop for Heap {
    fn drop(&mut self) {
        book().forget(self as *const Heap as usize);
        // A list in flight may still name it.
        #[cfg(windows)]
        if let Some(Some(heap)) = self.host.take() {
            crate::host::retire(heap);
        }
    }
}

/// Query types as `BeginQuery`, `EndQuery` and `ResolveQueryData` name them.
pub(crate) const OCCLUSION: u32 = 0;
pub(crate) const BINARY: u32 = 1;
pub(crate) const TIMESTAMP: u32 = 2;
pub(crate) const PIPELINE_STATISTICS: u32 = 3;

/// Which queries ran on the host, so a resolve reads only those. A query
/// is begun and ended on one host list; a title list cut at a wait
/// (`list::cut`) records into a new host list from there, so a query
/// still open at the cut is ended on the old one ([`Book::cut`]) and has
/// no host result until it runs whole again. Keys are (title heap, index);
/// lists are host lists by address.
#[derive(Default)]
struct Book {
    open: Vec<(usize, u32, usize)>,
    measured: std::collections::HashSet<(usize, u32)>,
}

impl Book {
    fn begin(&mut self, heap: usize, index: u32, list: usize) {
        self.open.retain(|&(h, i, _)| (h, i) != (heap, index));
        self.open.push((heap, index, list));
    }

    /// Whether to end the query on host list `list`: it was begun there
    /// (or is a timestamp, which has no begin).
    fn end(&mut self, heap: usize, index: u32, list: usize, timestamp: bool) -> bool {
        let begun = self.open.iter().position(|&(h, i, _)| (h, i) == (heap, index));
        let here = timestamp || begun.is_some_and(|at| self.open[at].2 == list);
        if let Some(at) = begun {
            self.open.swap_remove(at);
        }
        if here {
            self.measured.insert((heap, index));
        } else {
            self.measured.remove(&(heap, index));
        }
        here
    }

    /// Host list `list` closes at a cut: its open queries, to end on it
    /// now; they have no host result.
    fn cut(&mut self, list: usize) -> Vec<(usize, u32)> {
        let mut ended = Vec::new();
        self.open.retain(|&(h, i, l)| {
            if l == list {
                ended.push((h, i));
            }
            l != list
        });
        for key in &ended {
            self.measured.remove(key);
        }
        ended
    }

    fn measured(&self, heap: usize, index: u32) -> bool {
        self.measured.contains(&(heap, index))
    }

    fn forget(&mut self, heap: usize) {
        self.open.retain(|&(h, ..)| h != heap);
        self.measured.retain(|&(h, _)| h != heap);
    }
}

static BOOK: std::sync::LazyLock<std::sync::Mutex<Book>> = std::sync::LazyLock::new(Default::default);

fn book() -> std::sync::MutexGuard<'static, Book> {
    BOOK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Query types that run on the host: occlusion, binary occlusion, pipeline
/// statistics. `VICTORY_REAL_QUERIES=0`: none, every result plausible (as
/// before, to compare).
fn runs_on_host(kind: u32) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    matches!(kind, OCCLUSION | BINARY | PIPELINE_STATISTICS)
        && *ON.get_or_init(|| std::env::var_os("VICTORY_REAL_QUERIES").is_none_or(|v| v != "0"))
}

/// List methods `BeginQuery` (`begin`) and `EndQuery`, recorded on `list`.
#[cfg(windows)]
pub(crate) fn mark(list: &crate::host::HostList, heap: This, kind: u32, index: u32, begin: bool) {
    use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE;
    use windows::core::Interface;
    if !runs_on_host(kind) {
        return;
    }
    let Some(host_heap) = host_heap(heap) else { return };
    let at = list.list.as_raw() as usize;
    let mut book = book();
    if begin {
        book.begin(heap as usize, index, at);
        // SAFETY: an open list and a live heap of the query's type.
        unsafe { list.list.BeginQuery(host_heap, D3D12_QUERY_TYPE(kind as i32), index) };
    } else if book.end(heap as usize, index, at, kind == TIMESTAMP) {
        // SAFETY: as above; begun on this list.
        unsafe { list.list.EndQuery(host_heap, D3D12_QUERY_TYPE(kind as i32), index) };
    }
}

/// `list` closes at a cut: the queries still open on it are ended there
/// ([`Book::cut`]).
#[cfg(windows)]
pub(crate) fn cut(list: &crate::host::HostList) {
    use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE;
    use windows::core::Interface;
    let ended = book().cut(list.list.as_raw() as usize);
    for (heap, index) in ended {
        let heap = heap as This;
        // SAFETY: a live heap: a dropped one leaves the book.
        let kind = unsafe { &*heap.cast::<Heap>() }.desc.kind;
        // Heap type 0 holds occlusion queries, 2 statistics.
        let query = if kind == 0 { OCCLUSION } else { PIPELINE_STATISTICS };
        if let Some(h) = host_heap(heap) {
            // SAFETY: an open list; begun on it.
            unsafe { list.list.EndQuery(h, D3D12_QUERY_TYPE(query as i32), index) };
        }
    }
}

/// `first..first + count` in runs all measured or all not: (start,
/// length, measured).
fn runs(first: u32, count: u32, measured: impl Fn(u32) -> bool) -> Vec<(u32, u32, bool)> {
    let mut out: Vec<(u32, u32, bool)> = Vec::new();
    for i in first..first.saturating_add(count) {
        let m = measured(i);
        match out.last_mut() {
            Some((_, n, was)) if *was == m => *n += 1,
            _ => out.push((i, 1, m)),
        }
    }
    out
}

/// `ResolveQueryData` of queries that run on the host: the measured ones
/// are resolved on `list` into a read-back buffer that lands in `dst` (the
/// title's buffer, `count * bytes` long, checked by the caller) once the
/// list has run; the others are returned (start, length), for plausible
/// values written after the read-back. `None`: none ran on the host.
#[cfg(windows)]
pub(crate) fn resolve(
    list: &crate::host::HostList,
    heap: This,
    kind: u32,
    first: u32,
    count: u32,
    dst: *mut u8,
    bytes: u64,
) -> Option<Vec<(u32, u32)>> {
    use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE;
    if !runs_on_host(kind) || count == 0 {
        return None;
    }
    let host_heap = host_heap(heap)?;
    let parts = {
        let book = book();
        runs(first, count, |i| book.measured(heap as usize, i))
    };
    if parts.iter().all(|&(.., m)| !m) {
        return None;
    }
    let len = u64::from(count) * bytes;
    let buffer = crate::host::host()?.read_back_buffer(len)?;
    let mut plausible = Vec::new();
    for (start, n, measured) in parts {
        if measured {
            // SAFETY: an open list, a live heap, our buffer (kept by the
            // list until it is reset), inside it.
            unsafe {
                list.list.ResolveQueryData(
                    host_heap,
                    D3D12_QUERY_TYPE(kind as i32),
                    start,
                    n,
                    &buffer,
                    u64::from(start - first) * bytes,
                )
            };
        } else {
            plausible.push((start, n));
        }
    }
    list.defer_read_back(buffer, dst, [len as usize, 1, len as usize]);
    Some(plausible)
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_is_measured_when_begun_and_ended_on_one_list() {
        let mut b = Book::default();
        b.begin(1, 5, 100);
        assert!(b.end(1, 5, 100, false));
        assert!(b.measured(1, 5));
        // Ended on another host list (the title list was cut between).
        b.begin(1, 5, 100);
        assert!(!b.end(1, 5, 200, false));
        assert!(!b.measured(1, 5));
        // Never begun: not ended on the host; timestamps have no begin.
        assert!(!b.end(1, 6, 100, false));
        assert!(b.end(1, 7, 100, true));
        assert!(b.measured(1, 7));
    }

    #[test]
    fn a_cut_ends_the_open_queries_of_its_list() {
        let mut b = Book::default();
        b.begin(1, 0, 100);
        assert!(b.end(1, 0, 100, false));
        b.begin(1, 0, 100);
        b.begin(1, 1, 100);
        b.begin(2, 0, 300);
        let mut ended = b.cut(100);
        ended.sort();
        assert_eq!(ended, [(1, 0), (1, 1)]);
        assert!(!b.measured(1, 0));
        // The title's end after the cut is not recorded on the new list.
        assert!(!b.end(1, 1, 200, false));
        assert!(b.end(2, 0, 300, false));
        b.forget(2);
        assert!(!b.measured(2, 0));
    }

    #[test]
    fn resolves_split_into_measured_runs() {
        let measured = |i: u32| matches!(i, 3 | 4 | 6);
        assert_eq!(
            runs(2, 6, measured),
            [(2, 1, false), (3, 2, true), (5, 1, false), (6, 1, true), (7, 1, false)]
        );
        assert!(runs(0, 0, measured).is_empty());
    }
}
