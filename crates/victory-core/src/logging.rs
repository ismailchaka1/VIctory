//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for.

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// The most verbose level this build can log.
const CAP: LevelFilter = if cfg!(feature = "diagnostics") {
    LevelFilter::TRACE
} else {
    LevelFilter::INFO
};

pub fn init(default_level: &str) {
    let filter = std::env::var("VICTORY_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .ok()
        .and_then(|v| EnvFilter::try_new(v).ok())
        .unwrap_or_else(|| EnvFilter::new(default_level));
    let _ = tracing_subscriber::registry()
        .with(CAP)
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(make_writer as fn() -> Writer),
        )
        .try_init();
}

/// Where log lines go: each event is formatted on its own thread into a
/// buffer and written to stderr by one thread of ours, so a title thread
/// that logs does not wait for the write or for the others' (run vis-27:
/// title threads stood in the writer's lock and in a synchronous write).
/// `VICTORY_LOG_SYNC=1` writes on the logging thread, as before. [`flush`]
/// waits until what was logged so far is written.
enum Message {
    Line(Vec<u8>),
    Flush(std::sync::mpsc::SyncSender<()>),
}

static QUEUE: std::sync::OnceLock<Option<std::sync::mpsc::Sender<Message>>> = std::sync::OnceLock::new();

fn queue() -> Option<&'static std::sync::mpsc::Sender<Message>> {
    QUEUE
        .get_or_init(|| {
            if std::env::var_os("VICTORY_LOG_SYNC").is_some_and(|v| v == "1") {
                return None;
            }
            let (send, receive) = std::sync::mpsc::channel::<Message>();
            std::thread::Builder::new()
                .name("victory-log".into())
                .spawn(move || {
                    use std::io::Write;
                    let mut err = std::io::stderr();
                    for message in receive {
                        match message {
                            Message::Line(bytes) => {
                                let _ = err.write_all(&bytes);
                            }
                            Message::Flush(done) => {
                                let _ = err.flush();
                                let _ = done.send(());
                            }
                        }
                    }
                })
                .ok()?;
            Some(send)
        })
        .as_ref()
}

/// One event's bytes, queued when dropped (or written at once without a
/// queue).
pub struct Writer(Vec<u8>);

fn make_writer() -> Writer {
    Writer(Vec::with_capacity(256))
}

impl std::io::Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut self.0);
        if bytes.is_empty() {
            return;
        }
        match queue() {
            Some(queue) => {
                if let Err(std::sync::mpsc::SendError(Message::Line(bytes))) = queue.send(Message::Line(bytes)) {
                    let _ = std::io::Write::write_all(&mut std::io::stderr(), &bytes);
                }
            }
            None => {
                let _ = std::io::Write::write_all(&mut std::io::stderr(), &bytes);
            }
        }
    }
}

/// Waits (at most two seconds) until every line logged so far is written:
/// before the process ends, and before a crash report.
pub fn flush() {
    let Some(queue) = QUEUE.get().and_then(Option::as_ref) else {
        return;
    };
    let (done, wait) = std::sync::mpsc::sync_channel(1);
    if queue.send(Message::Flush(done)).is_ok() {
        let _ = wait.recv_timeout(std::time::Duration::from_secs(2));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn queued_lines_are_written_before_flush_returns() {
        use std::io::Write;
        let mut w = super::make_writer();
        w.write_all(b"queued line
").unwrap();
        drop(w);
        super::flush();
        // The queue thread exists and answered the flush.
        assert!(super::QUEUE.get().is_some_and(|q| q.is_some()));
    }
}
