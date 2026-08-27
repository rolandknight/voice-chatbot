use crate::config::SpeechUnlockType;
use crate::model::Detection;
use crate::oww;
use crate::oww::OwwModel;
use crate::oww::audio::AudioFeaturesTract;
use circular_buffer::CircularBuffer;
use log::{debug, trace, warn};
use oww::DETECTION_BUFFER_SIZE;
use rust_embed::Embed;
use std::io::Cursor;
use std::path::Path;
use std::time::Instant;
use std::{fs, io};
use tract_core::internal::TVec;
use tract_core::model::IntoRunnable;
use tract_core::prelude::multithread::{self, Executor};
use tract_core::prelude::{Framework, TValue};
use tract_onnx::prelude::{InferenceModelExt, IntoTensor, Tensor as TractTensor, tvec};

const MIN_POSITIVE_DETECTIONS: f32 = 2.0;
const NO_DETECTION_MS: u32 = 2_000;

#[derive(Embed)]
#[folder = "speech_models/"]
struct SpeechModels;

impl OwwModel {
    pub fn detection(&mut self, chunk_f32: Vec<f32>) -> Detection {
        let start = Instant::now();

        let Some(audio) = self.audio.as_mut() else {
            warn!("detection() on a frontend-less head; use detect(features)");
            return crate::model::Detection {
                detected: false,
                probability: 0.0,
                duration_ms: 0,
            };
        };
        let audio_features = match audio.get_audio_features(chunk_f32.as_slice()) {
            Ok(features) => features,
            Err(e) => {
                warn!("Embeddings error {:?}", e);
                return crate::model::Detection {
                    detected: false,
                    probability: 0.0,
                    duration_ms: 0,
                };
            }
        };

        let (detected, prc) = self.detect(audio_features);

        let onnx_duration = start.elapsed();
        Detection {
            detected,
            probability: prc,
            duration_ms: onnx_duration.as_millis(),
        }
    }

    pub fn detect(&mut self, features: TractTensor) -> (bool, f32) {
        trace!("2: features size {:?}", features.shape()); // [16, 96]
        let last = features.into_shape(&[1, 16, 96]).unwrap();
        trace!("2: inputs size {:?}", last.shape()); // [1, 16, 96]

        multithread::set_default_executor(Executor::SingleThread);

        let out: TVec<TValue> = self.tract_model.run(tvec!(last.into())).unwrap();
        trace!("2: output {:?}", out[0].shape()); // [1,1]

        let t = out.clone()[0]
            .clone()
            .into_tensor()
            .cast_to::<f32>()
            .unwrap()
            .into_owned();
        let probability = t.into_plain_array::<f32>().unwrap().as_slice().unwrap()[0];
        trace!("2:Tract probability: {:?}", probability);

        self.detections_buffer.push_back(probability);

        let average_detection_probability = self.calculate_average();
        let since_last_detection = self.last_detection_time.elapsed().as_millis();

        // Trigger when the smoothed average exceeds the threshold (rising-edge on average).
        // This matches the Python openWakeWord approach and avoids requiring a probability
        // drop below 0.1 (the old falling-edge), which often missed confident detections.
        if average_detection_probability > self.threshold
            && since_last_detection > NO_DETECTION_MS as _
        {
            self.last_detection_time = Instant::now();
            // Clear the buffer so the same utterance cannot retrigger before the refractory ends.
            self.detections_buffer.clear();
            return (true, average_detection_probability);
        }
        if average_detection_probability > 0.1 {
            debug!(
                "Prob {}, avg {} since {:?}",
                probability, average_detection_probability, since_last_detection
            );
        }
        (false, average_detection_probability)
    }

    fn calculate_average(&self) -> f32 {
        let all_detections = self.detections_buffer.to_vec();
        let mut detection_cumulative = 0.0f32;
        let mut positive_count = 0.0f32;
        for d in all_detections {
            if d > self.threshold {
                positive_count += 1.0;
                detection_cumulative += d;
            }
        }
        if positive_count < MIN_POSITIVE_DETECTIONS {
            return 0.0;
        }
        let avg = detection_cumulative / positive_count;
        if avg > self.threshold { avg } else { 0.0 }
    }

    pub fn new(model_type: SpeechUnlockType, threshold: f32) -> Result<OwwModel, String> {
        let model_data = match model_type {
            SpeechUnlockType::OpenWakeWordAlexa => {
                &crate::oww::oww_model::SpeechModels::get("alexa.onnx")
                    .unwrap()
                    .data
            }
            SpeechUnlockType::OpenWakeWordHeyMycroft => {
                &crate::oww::oww_model::SpeechModels::get("hey_mycroft_v0.1.onnx")
                    .unwrap()
                    .data
            }
            SpeechUnlockType::OpenWakeWordHeyJarvis => {
                &crate::oww::oww_model::SpeechModels::get("hey_jarvis_v0.1.onnx")
                    .unwrap()
                    .data
            }
            SpeechUnlockType::OpenWakeWordAhojHugo => {
                &crate::oww::oww_model::SpeechModels::get("ahoj_hugo.onnx")
                    .unwrap()
                    .data
            }
        };

        let model_unlock_word = match model_type {
            SpeechUnlockType::OpenWakeWordAlexa => "Alexa".to_string(),
            SpeechUnlockType::OpenWakeWordHeyMycroft => "Hey Mycroft".to_string(),
            SpeechUnlockType::OpenWakeWordHeyJarvis => "Hey Jarvis".to_string(),
            SpeechUnlockType::OpenWakeWordAhojHugo => "Ahoj Hugo".to_string(),
        };
        let detections_buffer = CircularBuffer::<DETECTION_BUFFER_SIZE, f32>::new();

        let mut rdr = Cursor::new(model_data);

        let tract_model = tract_onnx::onnx()
            .model_for_read(&mut rdr)
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        Ok(OwwModel {
            audio: Some(AudioFeaturesTract::create_default()),
            tract_model,
            threshold,
            last_detection_time: Instant::now(),
            detections_buffer,
            model_unlock_word,
        })
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
        model_unlock_word: String,
        threshold: f32,
    ) -> io::Result<OwwModel> {
        let model_data = fs::read(path)?;
        let detections_buffer = CircularBuffer::<DETECTION_BUFFER_SIZE, f32>::new();

        let mut rdr = Cursor::new(model_data);

        let tract_model = tract_onnx::onnx()
            .model_for_read(&mut rdr)
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        Ok(OwwModel {
            audio: Some(AudioFeaturesTract::create_default()),
            tract_model,
            threshold,
            last_detection_time: Instant::now(),
            detections_buffer,
            model_unlock_word,
        })
    }
}

impl OwwModel {
    /// Head-only model (no melspectrogram/embedding frontend): several heads
    /// can then share one [`AudioFeaturesTract`], each fed the same features
    /// through [`OwwModel::detect`]. `detection()` is not usable on it.
    pub fn head_from_path(path: &std::path::Path, threshold: f32) -> Result<OwwModel, String> {
        let mut m = Self::new_from_path_inner(path, threshold, false)?;
        // Start outside the refractory window: a freshly loaded head must be
        // able to fire at once, not 2 s after construction.
        m.last_detection_time = Instant::now()
            .checked_sub(std::time::Duration::from_millis(NO_DETECTION_MS as u64 + 1))
            .unwrap_or_else(Instant::now);
        m.model_unlock_word = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "custom".to_string());
        Ok(m)
    }

    /// PoC probe: build from an arbitrary openWakeWord head model on disk.
    pub fn new_from_path(path: &std::path::Path, threshold: f32) -> Result<OwwModel, String> {
        Self::new_from_path_inner(path, threshold, true)
    }

    fn new_from_path_inner(
        path: &std::path::Path,
        threshold: f32,
        with_frontend: bool,
    ) -> Result<OwwModel, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let mut rdr = Cursor::new(bytes);
        let tract_model = tract_onnx::onnx()
            .model_for_read(&mut rdr)
            .map_err(|e| e.to_string())?
            .into_optimized()
            .map_err(|e| e.to_string())?
            .into_runnable()
            .map_err(|e| e.to_string())?;
        Ok(OwwModel {
            audio: with_frontend.then(AudioFeaturesTract::create_default),
            tract_model: tract_model.into(),
            threshold,
            last_detection_time: Instant::now(),
            detections_buffer: CircularBuffer::new(),
            model_unlock_word: "custom".to_string(),
        })
    }
}

#[cfg(test)]
mod poc_probe {
    use super::*;
    #[test]
    fn probe_custom_model_on_wav() {
        let (Ok(model_path), Ok(wav)) = (std::env::var("POC_MODEL"), std::env::var("POC_WAV"))
        else {
            return;
        };
        let mut m = OwwModel::new_from_path(std::path::Path::new(&model_path), 0.3).unwrap();
        let mut reader = hound::WavReader::open(&wav).unwrap();
        let pcm: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32).collect();
        let mut maxp = 0f32;
        for chunk in pcm.chunks_exact(1280) {
            let d = m.detection(chunk.to_vec());
            if d.probability > maxp {
                maxp = d.probability;
            }
        }
        println!("oww_rs max probability: {maxp:.4}");
    }
}
