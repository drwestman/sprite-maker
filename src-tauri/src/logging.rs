use std::path::Path;

#[cfg(debug_assertions)]
const LOG_FILE_PREFIX: &str = "ollama.log.";
#[cfg(debug_assertions)]
const RETENTION_DAYS: i64 = 14;

#[cfg(debug_assertions)]
use chrono::{Duration, Local, NaiveDate};
#[cfg(debug_assertions)]
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};
#[cfg(debug_assertions)]
use tracing_appender::non_blocking::WorkerGuard;
#[cfg(debug_assertions)]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[cfg(debug_assertions)]
static LOG_GUARD: OnceLock<WorkerGuard> = OnceLock::new();
#[cfg(debug_assertions)]
static LOG_INIT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn initialize(data_directory: &Path) -> Result<(), String> {
    #[cfg(not(debug_assertions))]
    {
        let _ = data_directory;
        Ok(())
    }

    #[cfg(debug_assertions)]
    {
        let log_directory = data_directory.join("logs");
        fs::create_dir_all(&log_directory)
            .map_err(|error| format!("could not create {}: {error}", log_directory.display()))?;
        prune_old_logs(&log_directory, Local::now().date_naive())
            .map_err(|error| format!("could not prune {}: {error}", log_directory.display()))?;

        let _init_lock = LOG_INIT_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| "Ollama logger initialization lock was poisoned".to_string())?;
        if LOG_GUARD.get().is_some() {
            return Ok(());
        }
        let (writer, guard) =
            tracing_appender::non_blocking(LocalRollingWriter::new(&log_directory));
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_target(true)
                    .with_writer(writer),
            )
            .with(EnvFilter::new("off,ollama=debug"))
            .try_init()
            .map_err(|error| format!("could not initialize Ollama logger: {error}"))?;
        let _ = LOG_GUARD.set(guard);
        Ok(())
    }
}

#[cfg(debug_assertions)]
#[derive(Clone)]
struct LocalRollingWriter {
    state: Arc<Mutex<LocalRollingState>>,
}

#[cfg(debug_assertions)]
struct LocalRollingState {
    directory: PathBuf,
    date: Option<NaiveDate>,
    file: Option<File>,
}

#[cfg(debug_assertions)]
impl LocalRollingWriter {
    fn new(directory: &Path) -> Self {
        Self {
            state: Arc::new(Mutex::new(LocalRollingState {
                directory: directory.to_path_buf(),
                date: None,
                file: None,
            })),
        }
    }
}

#[cfg(debug_assertions)]
impl Write for LocalRollingWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("Ollama log writer lock was poisoned"))?;
        let today = Local::now().date_naive();
        if state.date != Some(today) || state.file.is_none() {
            prune_old_logs(&state.directory, today)?;
            let path = state.directory.join(format!("{LOG_FILE_PREFIX}{today}"));
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            state.date = Some(today);
            state.file = Some(file);
        }
        state
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("Ollama log file was not opened"))?
            .write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| io::Error::other("Ollama log writer lock was poisoned"))?;
        if let Some(file) = state.file.as_mut() {
            file.flush()?;
        }
        Ok(())
    }
}

#[cfg(debug_assertions)]
fn prune_old_logs(log_directory: &Path, today: NaiveDate) -> std::io::Result<()> {
    let cutoff = today - Duration::days(RETENTION_DAYS - 1);
    for entry in fs::read_dir(log_directory)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() {
            continue;
        }
        let Some(date) = log_date(&path) else {
            continue;
        };
        if date < cutoff {
            fs::remove_file(path)?;
        }
    }
    Ok(())
}

#[cfg(debug_assertions)]
fn log_date(path: &Path) -> Option<NaiveDate> {
    let name = path.file_name()?.to_str()?;
    let date = name.strip_prefix(LOG_FILE_PREFIX)?;
    NaiveDate::parse_from_str(date, "%Y-%m-%d").ok()
}

#[cfg(all(test, debug_assertions))]
mod tests {
    use super::{log_date, prune_old_logs, LocalRollingWriter, LOG_FILE_PREFIX, RETENTION_DAYS};
    use chrono::{Duration, Local};
    use std::{fs, io::Write, path::PathBuf};
    use uuid::Uuid;

    fn test_directory() -> PathBuf {
        let directory = std::env::temp_dir().join(format!("sprite-studio-logs-{}", Uuid::new_v4()));
        fs::create_dir_all(&directory).expect("test log directory should exist");
        directory
    }

    #[test]
    fn retains_the_configured_number_of_calendar_days() {
        let directory = test_directory();
        let today = Local::now().date_naive();
        let retained = today - Duration::days(RETENTION_DAYS - 1);
        let expired = retained - Duration::days(1);
        fs::write(
            directory.join(format!("{LOG_FILE_PREFIX}{retained}")),
            "retained",
        )
        .expect("retained log should write");
        fs::write(
            directory.join(format!("{LOG_FILE_PREFIX}{expired}")),
            "expired",
        )
        .expect("expired log should write");
        fs::write(directory.join("other.log"), "untouched").expect("other log should write");

        prune_old_logs(&directory, today).expect("log cleanup should succeed");

        assert!(directory
            .join(format!("{LOG_FILE_PREFIX}{retained}"))
            .exists());
        assert!(!directory
            .join(format!("{LOG_FILE_PREFIX}{expired}"))
            .exists());
        assert!(directory.join("other.log").exists());
        fs::remove_dir_all(directory).expect("test log directory should remove");
    }

    #[test]
    fn parses_only_daily_ollama_log_names() {
        let valid = PathBuf::from(format!("{LOG_FILE_PREFIX}2026-09-15"));
        assert!(log_date(&valid).is_some());
        assert!(log_date(&PathBuf::from("ollama.log")).is_none());
        assert!(log_date(&PathBuf::from("ollama.log.not-a-date")).is_none());
    }

    #[test]
    fn writes_to_the_local_calendar_date_file() {
        let directory = test_directory();
        let today = Local::now().date_naive();
        let mut writer = LocalRollingWriter::new(&directory);
        writer
            .write_all(b"ollama diagnostic")
            .expect("log entry should write");

        let log_path = directory.join(format!("{LOG_FILE_PREFIX}{today}"));
        assert_eq!(
            fs::read_to_string(log_path).expect("daily log should exist"),
            "ollama diagnostic"
        );
        fs::remove_dir_all(directory).expect("test log directory should remove");
    }
}
