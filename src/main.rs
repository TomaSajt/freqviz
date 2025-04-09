use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Stream,
};

use std::f64::consts::TAU;

use raylib::prelude::*;

use num_complex::{Complex64, ComplexFloat};

struct AppState {
    time: f64,
    kernel_size: usize,
    center: f64,
    range: f64,
    height_scale: f64,
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
    audio_stream: Stream,
    samples_per_sec: u32,
}

fn fft_helper(data: &Vec<Complex64>, sign: i32) -> Vec<Complex64> {
    let n = data.len();
    assert_eq!(n.count_ones(), 1);

    if n == 1 {
        return data.clone();
    }

    let evens = data.iter().step_by(2).cloned().collect::<Vec<_>>();
    let odds = data.iter().skip(1).step_by(2).cloned().collect::<Vec<_>>();

    let evens_res = fft_helper(&evens, sign);
    let odds_res = fft_helper(&odds, sign);

    let mut res = vec![Complex64::default(); n];
    for t in 0..(n / 2) {
        let w = (sign * t as i32) as f64 / n as f64 * TAU;
        let xd = Complex64::new(0.0, w).exp();

        res[t] = evens_res[t] + xd * odds_res[t];
        res[t + n / 2] = evens_res[t] - xd * odds_res[t];
    }

    return res;
}

pub fn fft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, -1)
        .iter()
        .map(|x| x / data.len() as f64)
        .collect();
}

pub fn ifft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, 1);
}

fn is_almost_same(a: Complex64, b: Complex64) -> bool {
    (a - b).abs() < 0.00000001
}

#[test]
fn test_fft() {
    let a: Vec<Complex64> = vec![1.0.into(), 10.0.into(), 100.0.into(), 1000.0.into()];
    let b = fft(&a);
    let c = ifft(&b);
    let d = a.into_iter().zip(c).all(|x| is_almost_same(x.0, x.1));
    assert!(d);
}

fn write_sample() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create("sine.wav", spec).unwrap();
    let amplitude = i16::MAX as f64;
    for t in (0..44100).map(|x| x as f64) {
        let sample = 1.0 * (t * 261.6256 / 44100.0 * TAU).sin();
        assert!(sample.abs() <= 1.0, "Sample is cutting off!");
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..44100).map(|x| x as f64) {
        let sample = 1.0 * (t * 329.6276 / 44100.0 * TAU).sin();
        assert!(sample.abs() <= 1.0, "Sample is cutting off!");
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..44100).map(|x| x as f64) {
        let sample = 1.0 * (t * 391.9954 / 44100.0 * TAU).sin();
        assert!(sample.abs() <= 1.0, "Sample is cutting off!");
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..44100).map(|x| x as f64) {
        let sample = 1.0 * (t * 261.6256 / 44100.0 * TAU).sin();
        assert!(sample.abs() <= 1.0, "Sample is cutting off!");
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..44100).map(|x| x as f64) {
        let sample =
            0.5 * (t * 261.6256 / 44100.0 * TAU).sin() + 0.5 * (t * 329.6276 / 44100.0 * TAU).sin();
        assert!(sample.abs() <= 1.0, "Sample is cutting off!");
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..(44100 * 3)).map(|x| x as f64) {
        let sample = 0.33 * (t * 261.6256 / 44100.0 * TAU).sin()
            + 0.33 * (t * 329.6276 / 44100.0 * TAU).sin()
            + 0.33 * (t * 391.9954 / 44100.0 * TAU).sin();
        assert!(sample.abs() <= 1.0, "Sample is cutting off!");
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
}

fn update_state(rl: &mut RaylibHandle, state: &mut AppState) {
    let dt = rl.get_frame_time() as f64;
    state.time += dt;

    if rl.is_key_pressed(KeyboardKey::KEY_D) {
        state.kernel_size *= 2;
    }
    if state.kernel_size > 1 && rl.is_key_pressed(KeyboardKey::KEY_A) {
        state.kernel_size /= 2;
    }
    if rl.is_key_down(KeyboardKey::KEY_UP) {
        state.range *= 2.0.powf(dt);
    }
    if rl.is_key_down(KeyboardKey::KEY_DOWN) {
        state.range /= 2.0.powf(dt);
    }
    if rl.is_key_down(KeyboardKey::KEY_W) {
        state.height_scale *= 5.0.powf(dt);
    }
    if rl.is_key_down(KeyboardKey::KEY_S) {
        state.height_scale /= 5.0.powf(dt);
    }
    if rl.is_key_down(KeyboardKey::KEY_RIGHT) {
        state.center += state.range * 0.5 * dt;
    }
    if rl.is_key_down(KeyboardKey::KEY_LEFT) {
        state.center -= state.range * 0.5 * dt;
    }
}

fn make_audio_stream(
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
) -> Result<(Stream, u32), anyhow::Error> {
    let host = cpal::default_host();

    // Set up the input device and stream with the default input config.
    let device = host
        .default_input_device()
        .expect("failed to find input device");

    println!("Default input device: {}", device.name()?);

    let config = device
        .default_input_config()
        .expect("Failed to get default input config");

    let samples_per_sec = config.config().sample_rate.0.clone();

    println!("Default input config: {:?}", config);

    assert_eq!(config.config().channels, 2);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                let mut deque = deque_mutex.lock().unwrap();
                // TODO: figure out if we need to skip per channel
                for sample in data.iter().step_by(2) {
                    deque.push_back(sample.clone() as f64);
                }
            },
            |err| {
                eprintln!("an error occurred on stream: {}", err);
            },
            None,
        )?,
        sample_format => {
            return Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}'"
            )))
        }
    };

    Ok((stream, samples_per_sec))
}

fn range_transform(l1: f64, r1: f64, l2: f64, r2: f64, val: f64) -> f64 {
    (r2 - l2) / (r1 - l1) * (val - l1) + l2
}

fn main() -> Result<(), anyhow::Error> {
    // write_sample();

    let deque_mutex = Arc::new(Mutex::new(VecDeque::<f64>::new()));
    let (audio_stream, samples_per_sec) = make_audio_stream(deque_mutex.clone())?;
    audio_stream.play()?;

    let mut state = AppState {
        time: 0.0,

        kernel_size: 2048,

        center: 600.0,
        range: 1200.0,
        height_scale: 1.0,

        deque_mutex,
        audio_stream,
        samples_per_sec,
    };

    let (mut rl, thread) = raylib::init().size(2000, 700).title("FreqViz").build();

    while !rl.window_should_close() {
        update_state(&mut rl, &mut state);

        let curr_sample_start = (state.time * (samples_per_sec as f64)) as usize;
        println!("curr_sample_start={curr_sample_start}");

        let kernel_size = state.kernel_size;

        let data = {
            let mut deque = state.deque_mutex.lock().unwrap();
            println!("{}", deque.len());
            // only keep as much data in the deque as we need
            while deque.len() > state.kernel_size {
                deque.pop_front();
            }
            // pad the data with zeroes if not enough data
            while deque.len() < state.kernel_size {
                deque.push_front(0.0);
            }
            deque
                .iter()
                .map(|x| Into::<Complex64>::into(x.clone()))
                .collect::<Vec<_>>()
        };

        let spinner_freqs: Vec<Complex64> = fft(&data);

        let mut real_freqs: Vec<(f64, f64)> = vec![(0.0, 0.0); state.kernel_size / 2 + 1];

        // we know that spinner_freqs[i]*exp(i/n*tau*t)+spinner_freqs[n-i]*exp(-i/n*tau*t) has to sum to a real number
        // this way we know that = spinner_freqs[i] is the complex conjugate of spinner_freqs[n-i]

        // we don't want negative amplitudes
        // if it happens, we just store a half-turn phaseshift instead

        real_freqs[0] = (spinner_freqs[0].abs(), spinner_freqs[0].arg());
        assert!(spinner_freqs[0].im().abs() < 0.001);

        real_freqs[kernel_size / 2] = (
            spinner_freqs[kernel_size / 2].abs(),
            spinner_freqs[kernel_size / 2].arg(),
        );
        assert!(spinner_freqs[kernel_size / 2].im().abs() < 0.001);

        for i in 1..(kernel_size / 2) {
            // (a+bi)*exp(wit)+(a-bi)*exp(-wit) = 2a*cos(wt)-2b*cos(wt) = 2*sqrt(a^2+b^2)*cos(wt+atan2(b,a))
            real_freqs[i] = (2.0 * spinner_freqs[i].abs(), spinner_freqs[i].arg());
        }

        let num_freqs = kernel_size / 2 + 1;

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);
        d.draw_text("Hello", 10, 10, 20, Color::BISQUE);

        let w = d.get_screen_width();
        let h = d.get_screen_height();

        for freq in [110.0, 220.0, 440.0, 880.0, 1760.0] {
            let x = range_transform(
                state.center - state.range / 2.0,
                state.center + state.range / 2.0,
                0.0,
                w as f64,
                freq,
            ) as i32;

            d.draw_line(x, 0, x, h, Color::BLUE);
            let text = freq.to_string() + "Hz";
            let font_size = 20;
            d.draw_text(
                &text,
                x - d.measure_text(&text, font_size) / 2,
                h - font_size - 10,
                font_size,
                Color::BLUEVIOLET,
            );
        }

        for x in 0..w {
            let prog = x as f64 / w as f64;
            let rot_per_sec = (state.center - state.range / 2.0) + state.range * prog;
            let rot_per_sample = rot_per_sec / samples_per_sec as f64;
            let fract_rot_per_sample = rot_per_sample * kernel_size as f64;
            let i = fract_rot_per_sample.floor() as i32;
            if i < 0 {
                d.draw_line(x as i32, 0, x as i32, h, Color::RED);
                continue;
            }
            let i = i as usize;
            if i >= num_freqs {
                d.draw_line(x as i32, 0, x as i32, h, Color::RED);
                continue;
            }

            d.draw_line(
                x as i32,
                0,
                x as i32,
                (real_freqs[i].0 * h as f64 * state.height_scale) as i32,
                Color::BLACK,
            );
        }
    }

    Ok(())
}
