use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub struct DownloadProgress {
    multi: MultiProgress,
}

impl DownloadProgress {
    pub fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
        }
    }

    pub fn add_file(&self, name: &str, total_bytes: u64) -> ProgressBar {
        let style = ProgressStyle::with_template(
            "  {prefix:.bold} {bar:40.cyan/blue} {percent}% {bytes}/{total_bytes} {bytes_per_sec}",
        )
        .unwrap()
        .progress_chars("##-");

        let pb = self.multi.add(ProgressBar::new(total_bytes));
        pb.set_style(style);
        pb.set_prefix(name.to_string());
        pb
    }

    #[allow(dead_code)]
    pub fn add_spinner(&self, message: &str) -> ProgressBar {
        let style = ProgressStyle::with_template("  {spinner:.green} {msg}").unwrap();
        let pb = self.multi.add(ProgressBar::new_spinner());
        pb.set_style(style);
        pb.set_message(message.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(120));
        pb
    }
}
