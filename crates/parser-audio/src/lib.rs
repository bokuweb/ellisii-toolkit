//! Speech-to-text parser (Meeting Recorder Phase 2)。
//!
//! ## 設計
//!
//! - `Transcriber` trait で STT 実装を抽象化。テスト / wiring 用に
//!   `EchoTranscriber` (= input wav の filename を本文として返す stub) を同梱、
//!   real STT は `whisper-rs` 経由の `WhisperCppTranscriber` を feature
//!   `whispercpp` でビルドした時のみ提供する。
//! - `parse_audio(path, transcriber)` が wav (16-bit PCM) を読み、`Transcript`
//!   を経由して `ParsedDocument` を返す。後段の chunker は既存の
//!   `DefaultChunker` でそのまま処理する (segment 単位を 1 block にし、
//!   `heading_path` に "Audio / <ファイル名>" を積む)。
//! - **timestamp** は今はメタ情報として `Transcript::segments` に保持するのみ。
//!   `ParsedBlock` への持ち越しは Phase 3 (citation で動画位置ジャンプ) で行う
//!   際にスキーマ拡張する。
//!
//! ## なぜ trait 越し
//!
//! - whisper.cpp は重い C++ 依存 (cmake、~1-2 分ビルド)。CI / 軽量ビルドでは
//!   skip したい
//! - 将来 OpenAI Whisper API / Vosk / faster-whisper 等の他バックエンドを
//!   差し替え可能にする
//! - テストでは決定的な stub を渡したい
//!
//! ## 制限 (Phase 2 時点)
//!
//! - **wav 16-bit PCM 限定**: mp3 / m4a / flac / ogg は `SourceKind` 上は
//!   audio と判定されるが、本 crate では decode しない。recorder crate が
//!   出力する wav が想定 input。
//! - **sliding-window streaming は別関数**: 本 crate の `parse_audio` はバッチ
//!   経路で 1 度に全 audio を処理する。recorder の `SampleSink` 経由の live
//!   stream は `Transcriber::transcribe_chunk` (将来追加) で扱う想定。

use async_trait::async_trait;
use ellisii_core::SourceKind;
use ellisii_parsers_core::{ParsedBlock, ParsedDocument};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("wav: {0}")]
    Wav(#[from] hound::Error),
    #[error("transcriber: {0}")]
    Transcriber(String),
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// 文字起こし 1 セグメント。`start_ms` / `end_ms` は audio 開始からの経過時刻。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// 1 wav 分の文字起こし結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    /// BCP-47 風の言語タグ (例: `"ja"`, `"en"`, `"auto"`)。stub は `"unknown"`。
    pub language: String,
    pub segments: Vec<Segment>,
}

impl Transcript {
    /// 全 segment の text を改行で繋ぎ 1 つの大きな本文文字列にする。chunker は
    /// この本文を受け取って通常の段落単位 split を行う。
    pub fn body(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[async_trait]
pub trait Transcriber: Send + Sync {
    /// wav (16-bit PCM) ファイルを文字起こしする。`lang_hint` は `"ja"` などで
    /// 言語固定 (None なら auto detect)。
    async fn transcribe(&self, wav: &Path, lang_hint: Option<&str>) -> Result<Transcript>;
}

/// wav の filename を擬似 transcript として返す stub。CI 動作確認・wiring テスト用。
/// 実 STT が無い環境でも `parse_audio` のパイプライン全体を検証できる。
#[derive(Debug, Default)]
pub struct EchoTranscriber;

#[async_trait]
impl Transcriber for EchoTranscriber {
    async fn transcribe(&self, wav: &Path, _lang_hint: Option<&str>) -> Result<Transcript> {
        // wav ヘッダから duration を推定して 1 segment にする
        let reader = hound::WavReader::open(wav)?;
        let spec = reader.spec();
        let total_samples = reader.len() as u64;
        let frames = total_samples / spec.channels.max(1) as u64;
        let duration_ms = (frames as f64 * 1000.0 / spec.sample_rate.max(1) as f64) as u64;
        let stub_text = format!(
            "[EchoTranscriber stub] {} ({} ch @ {} Hz, {} ms)",
            wav.file_name().and_then(|s| s.to_str()).unwrap_or("audio"),
            spec.channels,
            spec.sample_rate,
            duration_ms
        );
        Ok(Transcript {
            language: "unknown".to_string(),
            segments: vec![Segment {
                start_ms: 0,
                end_ms: duration_ms,
                text: stub_text,
            }],
        })
    }
}

/// wav を `Transcriber` で文字起こし → `ParsedDocument` に変換する。
///
/// - `heading_path` には `["Audio", "<ファイル名>"]` を 1 段積む (chunker と
///   caption rerank が後で利用)
/// - 各 segment を別 `ParsedBlock` にする (chunker 側の自然な splitter として
///   段落単位を保つため)
/// - segment text が空白だけのものは弾く
pub async fn parse_audio(
    path: &Path,
    transcriber: &dyn Transcriber,
    lang_hint: Option<&str>,
) -> Result<ParsedDocument> {
    let transcript = transcriber.transcribe(path, lang_hint).await?;
    let file_label = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("audio")
        .to_string();
    let blocks: Vec<ParsedBlock> = transcript
        .segments
        .into_iter()
        .filter(|s| !s.text.trim().is_empty())
        .map(|s| ParsedBlock {
            text: s.text,
            heading_path: vec!["Audio".into(), file_label.clone()],
            // `page` を「セグメント開始秒」に流用するのは恣意的なので Phase 2
            // では持ち越さない。Phase 3 で timestamp_ms フィールドを
            // ParsedBlock に正式追加する想定。
            page: None,
            bbox: None,
        })
        .collect();
    Ok(ParsedDocument {
        blocks,
        kind: SourceKind::Audio,
    })
}

// ─── whisper.cpp バックエンド (feature = "whispercpp") ────────────────────

#[cfg(feature = "whispercpp")]
mod whispercpp_backend {
    use super::*;
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    /// whisper.cpp の Rust binding を使った STT 実装。
    pub struct WhisperCppTranscriber {
        ctx: WhisperContext,
        n_threads: i32,
    }

    impl WhisperCppTranscriber {
        /// GGUF / ggml モデルをロード。`n_threads` を None にすると CPU 物理コア
        /// 数 / 2 を採用 (cpu 過負荷を避けつつスループット維持)。
        pub fn load(model_path: &Path, n_threads: Option<i32>) -> Result<Self> {
            let params = WhisperContextParameters::default();
            let ctx = WhisperContext::new_with_params(
                model_path
                    .to_str()
                    .ok_or_else(|| Error::Transcriber("non-utf8 model path".into()))?,
                params,
            )
            .map_err(|e| Error::Transcriber(format!("whisper load: {e}")))?;
            let n_threads = n_threads.unwrap_or_else(|| {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get() as i32)
                    .unwrap_or(4);
                (cores / 2).max(1)
            });
            Ok(Self { ctx, n_threads })
        }
    }

    #[async_trait]
    impl Transcriber for WhisperCppTranscriber {
        async fn transcribe(&self, wav: &Path, lang_hint: Option<&str>) -> Result<Transcript> {
            // hound で wav を読み込み、16kHz mono f32 に正規化して whisper に渡す。
            let reader = hound::WavReader::open(wav)?;
            let spec = reader.spec();
            let samples: Vec<i16> = reader
                .into_samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Error::Wav)?;
            // channel interleaved → mono に平均化
            let mono: Vec<f32> = samples
                .chunks(spec.channels.max(1) as usize)
                .map(|frame| {
                    let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                    let avg = sum as f32 / frame.len() as f32;
                    avg / i16::MAX as f32
                })
                .collect();
            // 16kHz リサンプル (whisper の要求)
            let resampled = if spec.sample_rate == 16_000 {
                mono
            } else {
                linear_resample(&mono, spec.sample_rate, 16_000)
            };

            let mut state = self
                .ctx
                .create_state()
                .map_err(|e| Error::Transcriber(format!("create_state: {e}")))?;
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_n_threads(self.n_threads);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_special(false);
            params.set_print_timestamps(false);
            if let Some(lang) = lang_hint {
                params.set_language(Some(lang));
            } else {
                params.set_language(Some("auto"));
            }

            state
                .full(params, &resampled)
                .map_err(|e| Error::Transcriber(format!("full: {e}")))?;

            let num_segments = state
                .full_n_segments()
                .map_err(|e| Error::Transcriber(format!("n_segments: {e}")))?;
            let mut segments = Vec::with_capacity(num_segments as usize);
            for i in 0..num_segments {
                let text = state
                    .full_get_segment_text(i)
                    .map_err(|e| Error::Transcriber(format!("seg_text: {e}")))?;
                let t0 = state
                    .full_get_segment_t0(i)
                    .map_err(|e| Error::Transcriber(format!("seg_t0: {e}")))?;
                let t1 = state
                    .full_get_segment_t1(i)
                    .map_err(|e| Error::Transcriber(format!("seg_t1: {e}")))?;
                // whisper-rs の t0/t1 は 10ms 単位なので ×10 で ms に直す
                segments.push(Segment {
                    start_ms: (t0 as u64) * 10,
                    end_ms: (t1 as u64) * 10,
                    text: text.trim().to_string(),
                });
            }
            let language = lang_hint.unwrap_or("auto").to_string();
            Ok(Transcript { language, segments })
        }
    }

    /// 単純な線形補間リサンプラ (whisper 入力用 16kHz への変換)。
    /// 高品質には rubato を使うべきだが、whisper 入力ではこの粗さでも十分
    /// (whisper 自体が log-mel に変換する過程で smoothing がかかる)。
    fn linear_resample(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
        if input.is_empty() || from_hz == to_hz {
            return input.to_vec();
        }
        let ratio = to_hz as f64 / from_hz as f64;
        let out_len = (input.len() as f64 * ratio) as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f64 / ratio;
            let idx = src.floor() as usize;
            let frac = src - idx as f64;
            let a = input[idx.min(input.len() - 1)];
            let b = input[(idx + 1).min(input.len() - 1)];
            out.push(a + (b - a) * frac as f32);
        }
        out
    }
}

#[cfg(feature = "whispercpp")]
pub use whispercpp_backend::WhisperCppTranscriber;

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-bit PCM の 1 秒分 wav を作るヘルパ
    fn write_silent_wav(path: &Path, sample_rate: u32, duration_secs: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        let total = (sample_rate as f32 * duration_secs) as usize;
        for _ in 0..total {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[tokio::test]
    async fn echo_transcriber_returns_filename_stub_with_duration() {
        let tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        write_silent_wav(tmp.path(), 16_000, 0.5);
        let t = EchoTranscriber;
        let r = t.transcribe(tmp.path(), None).await.unwrap();
        assert_eq!(r.segments.len(), 1);
        assert_eq!(r.language, "unknown");
        assert!(r.segments[0].text.contains("EchoTranscriber stub"));
        assert!(r.segments[0].text.contains("16000 Hz"));
        // 500 ms 録音なので duration_ms は 500 前後
        assert_eq!(r.segments[0].start_ms, 0);
        assert!(r.segments[0].end_ms >= 400 && r.segments[0].end_ms <= 600);
    }

    #[tokio::test]
    async fn parse_audio_produces_one_block_per_nonempty_segment() {
        let tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        write_silent_wav(tmp.path(), 16_000, 0.3);
        let t = EchoTranscriber;
        let doc = parse_audio(tmp.path(), &t, None).await.unwrap();
        assert_eq!(doc.kind, SourceKind::Audio);
        assert_eq!(doc.blocks.len(), 1);
        let block = &doc.blocks[0];
        assert_eq!(block.heading_path.len(), 2);
        assert_eq!(block.heading_path[0], "Audio");
        assert!(block.heading_path[1].ends_with(".wav"));
        assert!(block.text.contains("EchoTranscriber stub"));
    }

    #[tokio::test]
    async fn parse_audio_drops_whitespace_only_segments() {
        struct WhiteOnly;
        #[async_trait]
        impl Transcriber for WhiteOnly {
            async fn transcribe(&self, _wav: &Path, _lang: Option<&str>) -> Result<Transcript> {
                Ok(Transcript {
                    language: "ja".into(),
                    segments: vec![
                        Segment {
                            start_ms: 0,
                            end_ms: 100,
                            text: "  ".into(),
                        },
                        Segment {
                            start_ms: 100,
                            end_ms: 200,
                            text: "本文あり".into(),
                        },
                        Segment {
                            start_ms: 200,
                            end_ms: 300,
                            text: "\n\t".into(),
                        },
                    ],
                })
            }
        }
        let tmp = tempfile::NamedTempFile::with_suffix(".wav").unwrap();
        write_silent_wav(tmp.path(), 16_000, 0.1);
        let doc = parse_audio(tmp.path(), &WhiteOnly, None).await.unwrap();
        assert_eq!(doc.blocks.len(), 1);
        assert_eq!(doc.blocks[0].text, "本文あり");
    }

    #[test]
    fn transcript_body_joins_segments_with_newline() {
        let t = Transcript {
            language: "ja".into(),
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 100,
                    text: "A".into(),
                },
                Segment {
                    start_ms: 100,
                    end_ms: 200,
                    text: "B".into(),
                },
            ],
        };
        assert_eq!(t.body(), "A\nB");
    }
}
