use std::{
    collections::VecDeque,
    fs::File,
    io::BufReader,
    path::Path,
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

use cpal::traits::{DeviceTrait, HostTrait};
use ringbuf::{
    storage::Heap,
    traits::{Consumer, Producer, Split},
    HeapRb,
};

pub fn make_microphone_input_audio_stream(
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
) -> Result<(cpal::Stream, u32), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_input_device()
        .expect("Failed to find input device");

    let config = device
        .default_input_config()
        .expect("Failed to get default input config");

    fn part2<T>(
        device: cpal::Device,
        config: cpal::SupportedStreamConfig,
        deque_mutex: Arc<Mutex<VecDeque<f64>>>,
    ) -> Result<(cpal::Stream, u32), anyhow::Error>
    where
        T: cpal::SizedSample,
        f64: cpal::FromSample<T>,
    {
        let mut curr_channel = 0;
        let channels = config.channels();
        let sample_rate = config.sample_rate().0;

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[T], _: &_| {
                assert_eq!(data.len() % channels as usize, 0);
                let mut deque = deque_mutex.lock().unwrap();
                for sample in data {
                    if curr_channel == 0 {
                        let val: f64 = cpal::Sample::from_sample(*sample);
                        deque.push_back(val);
                    }
                    curr_channel = (curr_channel + 1) % channels;
                }
            },
            |err| {
                eprintln!("an error occurred on stream: {}", err);
            },
            None,
        )?;

        Ok((stream, sample_rate))
    }

    match config.sample_format() {
        cpal::SampleFormat::I8 => part2::<i8>(device, config, deque_mutex),
        cpal::SampleFormat::I16 => part2::<i16>(device, config, deque_mutex),
        cpal::SampleFormat::I32 => part2::<i32>(device, config, deque_mutex),
        cpal::SampleFormat::I64 => part2::<i64>(device, config, deque_mutex),
        cpal::SampleFormat::U8 => part2::<u8>(device, config, deque_mutex),
        cpal::SampleFormat::U16 => part2::<u16>(device, config, deque_mutex),
        cpal::SampleFormat::U32 => part2::<u32>(device, config, deque_mutex),
        cpal::SampleFormat::U64 => part2::<u64>(device, config, deque_mutex),
        cpal::SampleFormat::F32 => part2::<f32>(device, config, deque_mutex),
        cpal::SampleFormat::F64 => part2::<f64>(device, config, deque_mutex),
        _ => unreachable!(),
    }
}

pub fn make_output_audio_stream(
    path: &Path,
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
) -> Result<(cpal::Stream, JoinHandle<()>, u32), anyhow::Error> {
    let reader = hound::WavReader::open(path)?;
    let channels = reader.spec().channels;
    let sample_rate = reader.spec().sample_rate;
    let sample_format_type = reader.spec().sample_format;
    let bits_per_sample = reader.spec().bits_per_sample;

    let sample_format = match sample_format_type {
        hound::SampleFormat::Float => match bits_per_sample {
            32 => cpal::SampleFormat::F32,
            64 => cpal::SampleFormat::F64,
            _ => unreachable!(),
        },
        hound::SampleFormat::Int => match bits_per_sample {
            8 => cpal::SampleFormat::I8,
            16 => cpal::SampleFormat::I16,
            32 => cpal::SampleFormat::I32,
            _ => unreachable!(),
        },
    };

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("Failed to find input device");

    let config = device
        .supported_output_configs()
        .expect("Failed to get supported config ranges")
        .filter(|x| x.sample_format() == sample_format && x.channels() == channels)
        .find_map(|x| x.try_with_sample_rate(cpal::SampleRate(sample_rate)))
        .expect("No supported config found for the audio file");

    fn part2<T>(
        mut reader: hound::WavReader<BufReader<File>>,
        device: cpal::Device,
        supported_config: cpal::SupportedStreamConfig,
        channels: u16,
        deque_mutex: Arc<Mutex<VecDeque<f64>>>,
    ) -> Result<(cpal::Stream, JoinHandle<()>, u32), anyhow::Error>
    where
        T: hound::Sample + cpal::SizedSample + 'static,
        f64: cpal::FromSample<T>,
        Heap<T>: Sync + Send,
    {
        let mut curr_channel = 0;

        let cpal::SampleRate(sample_rate) = supported_config.sample_rate();

        let rb = HeapRb::<T>::new(sample_rate as usize * channels as usize); // 1 second buffer
        let (mut prod, mut cons) = rb.split();

        // Spawn thread to stream data from file into buffer
        let read_thread_handle = thread::spawn(move || {
            for sample in reader.samples::<T>() {
                let val = sample.unwrap();
                while !prod.try_push(val).is_ok() {} // wait until there's space in the buffer
            }
        });

        let stream = device.build_output_stream(
            &supported_config.into(),
            move |data: &mut [T], _: &_| {
                assert_eq!(data.len() % channels as usize, 0);
                let mut deque = deque_mutex.lock().unwrap();
                for sample in data.as_mut() {
                    *sample = cons.try_pop().unwrap_or(cpal::Sample::EQUILIBRIUM);
                    if curr_channel == 0 {
                        let val: f64 = cpal::Sample::from_sample(*sample);
                        deque.push_back(val);
                    }
                    curr_channel = (curr_channel + 1) % channels;
                }
            },
            |err| eprintln!("An error occurred on the output audio stream: {}", err),
            None,
        )?;

        Ok((stream, read_thread_handle, sample_rate))
    }

    match sample_format {
        cpal::SampleFormat::I8 => part2::<i8>(reader, device, config, channels, deque_mutex),
        cpal::SampleFormat::I16 => part2::<i16>(reader, device, config, channels, deque_mutex),
        cpal::SampleFormat::I32 => part2::<i32>(reader, device, config, channels, deque_mutex),
        cpal::SampleFormat::F32 => part2::<f32>(reader, device, config, channels, deque_mutex),
        _ => unreachable!(),
    }
}
