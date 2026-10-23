//! Loading point clouds while they are rendered.
//!
//! A producer (a thread on native, the `fetch` loop of the page on the web) feeds the bytes of
//! a ply file to a [`PlyFeeder`], which parses them and puts the converted points into a
//! [`PointCloudStream`]. The renderer takes the points out of the stream and uploads them to the
//! GPU.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use super::ply::{GaussianChunk, PlyInfo, PlyStream};

#[derive(Default)]
struct Shared {
    info: Option<PlyInfo>,
    chunks: VecDeque<GaussianChunk>,
    queued_points: usize,
    finished: bool,
    error: Option<String>,
}

/// The consumer side of a point cloud that is being loaded. Cheap to clone, all clones share the
/// same queue.
#[derive(Clone, Default)]
pub struct PointCloudStream(Arc<Mutex<Shared>>);

impl PointCloudStream {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// available as soon as the ply header was received
    pub fn info(&self) -> Option<PlyInfo> {
        self.lock().info.clone()
    }

    /// the error that stopped loading, if any
    pub fn error(&self) -> Option<String> {
        self.lock().error.clone()
    }

    /// no more points will be added to the queue (all points were read or loading failed)
    pub fn is_finished(&self) -> bool {
        let shared = self.lock();
        shared.finished || shared.error.is_some()
    }

    /// number of points that are waiting to be taken
    pub fn queued_points(&self) -> usize {
        self.lock().queued_points
    }

    /// Takes queued chunks with about `max_points` points in total (but at least one chunk, if
    /// there is any).
    pub fn take(&self, max_points: usize) -> Vec<GaussianChunk> {
        let mut shared = self.lock();
        let mut taken = Vec::new();
        let mut points = 0;
        while points < max_points {
            let Some(chunk) = shared.chunks.pop_front() else {
                break;
            };
            points += chunk.len();
            taken.push(chunk);
        }
        shared.queued_points -= points;
        taken
    }

    /// whether the producer is gone, which means that nobody is interested in more data
    #[cfg(not(target_arch = "wasm32"))]
    fn is_abandoned(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }
}

/// The producer side of a [`PointCloudStream`].
pub struct PlyFeeder {
    stream: PointCloudStream,
    parser: PlyStream,
}

impl PlyFeeder {
    pub fn new() -> Self {
        Self {
            stream: PointCloudStream::default(),
            parser: PlyStream::new(),
        }
    }

    pub fn stream(&self) -> PointCloudStream {
        self.stream.clone()
    }

    /// whether the header was received
    pub fn is_ready(&self) -> bool {
        self.parser.info().is_some()
    }

    /// Parses the next bytes of the file. On failure the error is also reported through the
    /// stream.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), anyhow::Error> {
        match self.parser.push(bytes) {
            Ok(chunk) => {
                let mut shared = self.stream.lock();
                if shared.info.is_none() {
                    shared.info = self.parser.info().cloned();
                }
                if let Some(chunk) = chunk.filter(|c| c.len() > 0) {
                    shared.queued_points += chunk.len();
                    shared.chunks.push_back(chunk);
                }
                shared.finished = self.parser.is_complete();
                Ok(())
            }
            Err(err) => {
                self.stream.lock().error = Some(format!("{err:#}"));
                Err(err)
            }
        }
    }

    /// to be called when there is no more data
    pub fn finish(&mut self) -> Result<(), anyhow::Error> {
        if self.parser.is_complete() {
            return Ok(());
        }
        let err = anyhow::anyhow!(
            "unexpected end of file: read {} of {} points",
            self.parser.loaded_points(),
            self.parser.info().map_or(0, |i| i.num_points)
        );
        self.stream.lock().error = Some(format!("{err:#}"));
        Err(err)
    }
}

/// at most this many converted points wait to be uploaded, the reader waits for the renderer
/// when it is faster
#[cfg(not(target_arch = "wasm32"))]
const MAX_QUEUED_POINTS: usize = 1 << 21;

/// Reads a ply file on a separate thread. Returns when the header was read, so
/// [`PointCloudStream::info`] is available.
#[cfg(not(target_arch = "wasm32"))]
pub fn spawn_reader<R: std::io::Read + Send + 'static>(
    mut reader: R,
) -> Result<PointCloudStream, anyhow::Error> {
    use std::time::Duration;

    let mut feeder = PlyFeeder::new();
    let stream = feeder.stream();
    std::thread::Builder::new()
        .name("ply loader".to_string())
        .spawn(move || {
            let mut buf = vec![0u8; 8 << 20];
            while !feeder.stream.is_finished() {
                if feeder.stream.is_abandoned() {
                    return;
                }
                if feeder.stream.queued_points() > MAX_QUEUED_POINTS {
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                }
                match reader.read(&mut buf) {
                    Ok(0) => {
                        feeder.finish().ok();
                        return;
                    }
                    Ok(n) => {
                        if feeder.push(&buf[..n]).is_err() {
                            return;
                        }
                    }
                    Err(err) => {
                        feeder.stream.lock().error = Some(err.to_string());
                        return;
                    }
                }
            }
        })?;

    while stream.info().is_none() {
        if let Some(err) = stream.error() {
            anyhow::bail!(err);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(stream)
}
