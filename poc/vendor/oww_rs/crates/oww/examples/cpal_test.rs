use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, info, warn};
use std::process::exit;
use std::sync::{Arc, Mutex};
use oww_rs::config::SpeechUnlockType::{OpenWakeWordAlexa};
use audio_tools::mic_config::find_best_config;
use audio_tools::process_audio::resample_into_chunks;
use audio_tools::resampler::make_resampler;
use oww_rs::oww::{OwwModel, OWW_MODEL_CHUNK_SIZE};

use oww_rs::mic_cpal::build_input_stream;
use std::sync::mpsc;
use std::thread;

/// Demonstrating the lib usage with cpal streaming from microphone. The code here uses cpal
/// to connect to different data types from microphones and trying to convert it to model's
/// required format that is sample rate 16kHz and f32 data format
fn main() -> Result<(), anyhow::Error> {
    env_logger::Builder::new().filter_level(log::LevelFilter::Info).init();
    // Initialize CPAL
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    match device.description() {
        Ok(desc) => {
            debug!("Input device: {}", desc.name());
        }
        Err(e) => {
            warn!("Couldn't get mic device: {:?}", e);
            exit(1);
        }
    }

    let (config, sample_format) = find_best_config(&device, false).unwrap();
    info!("Selected input config: {:?}", config);

    // Create a buffer to store audio data
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(vec![]));
    let buffer_clone = buffer.clone();

    // Store the original sample rate and channels
    let original_sample_rate = config.sample_rate as f32;
    let channels = config.channels as usize;

    // Create the input stream
    let err_fn = |err| warn!("An error occurred on the input stream: {}", err);

    let mut model = OwwModel::new(OpenWakeWordAlexa, 0.1).unwrap();

    let mut resampler = make_resampler(original_sample_rate as _, OWW_MODEL_CHUNK_SIZE as _, channels).unwrap();

    let (tx, rx) = mpsc::sync_channel(100);

    thread::spawn(move || {
        while let Ok(chunk) = rx.recv() {
            let d = model.detection(chunk);
            if d.detected {
                println!("Result {:?}", d);
            }
        }
    });

    let stream = build_input_stream(
        &device,
        &config,
        sample_format,
        move |data| {
            let chunks = resample_into_chunks(data, &buffer_clone, channels, &mut resampler);
            for chunk in chunks {
                if let Err(_) = tx.try_send(chunk.data_f32.first().clone()) {
                    warn!("Worker channel full, dropping chunk");
                }
            }
        },
        err_fn,
        None,
    ).unwrap();

    stream.play()?;

    println!("Listening for word `Alexa`");
    println!("Recording and resampling to 16000 Hz... Press Enter to stop.");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    Ok(())
}
