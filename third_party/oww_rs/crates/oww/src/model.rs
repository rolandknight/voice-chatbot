pub(crate) trait Model: Send + Sync {
    fn detect(&mut self, data: Vec<f32>) -> Option<Detection>;
    fn detect_i16(&mut self, data: Vec<i16>) -> Option<Detection>;
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub detected: bool,
    pub probability: f32,
    pub duration_ms: u128,
}

impl Detection {
    pub fn none() -> Detection {
        Detection {
            detected: false,
            probability: 0.0,
            duration_ms: 0,
        }
    }
}
