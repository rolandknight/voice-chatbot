use crate::ModelType;
use crate::model::{Detection, Model};
use crate::oww::audio::AudioFeaturesTract;
use circular_buffer::CircularBuffer;
use std::time::Instant;

pub mod audio;
mod oww_model;

pub const OWW_MODEL_CHUNK_SIZE: usize = 1280;

const DETECTION_BUFFER_SIZE: usize = 12; // 12 detection ~ 1 secs

#[derive(Debug)]
pub struct OwwModel {
    /// Melspectrogram + embedding frontend. `None` for a head built with
    /// [`OwwModel::head_from_path`], which is driven from a shared frontend
    /// through [`OwwModel::detect`].
    audio: Option<AudioFeaturesTract>,
    pub tract_model: ModelType,
    threshold: f32,
    pub last_detection_time: Instant,
    // pub detections_buffer: CircularBuffer<DETECTION_BUFFER_SIZE, f32>,
    pub detections_buffer: CircularBuffer<DETECTION_BUFFER_SIZE, f32>,
    pub model_unlock_word: String,
}

impl Model for OwwModel {
    fn detect(&mut self, chunk: Vec<f32>) -> Option<Detection> {
        // let chunk_f32 = chunk.to_vec().iter().map(|x| *x as f32).collect::<Vec<f32>>();
        Some(self.detection(chunk))
    }

    fn detect_i16(&mut self, _data: Vec<i16>) -> Option<Detection> {
        None
    }
}
