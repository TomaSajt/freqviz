use std::f64::consts::TAU;

use num_traits::Float;
use raylib::prelude::*;

use num_complex::{c64, Complex, Complex64, ComplexFloat};

fn fft_helper<T>(data: &Vec<Complex<T>>, sign: i32) -> Vec<Complex<T>>
where
    T: Copy + Float + Default + From<i32> + From<f64>,
{
    let n = data.len();
    assert_eq!(n.count_ones(), 1);

    if n == 1 {
        return data.clone();
    }

    let evens: Vec<Complex<T>> = data.iter().step_by(2).cloned().collect();
    let odds: Vec<Complex<T>> = data.iter().skip(1).step_by(2).cloned().collect();

    let evens_res = fft_helper(&evens, sign);
    let odds_res = fft_helper(&odds, sign);

    let mut res: Vec<Complex<T>> = vec![Complex::default(); n];
    for t in 0..(n / 2) {
        let val1: T = (sign * t as i32).into();
        let val2: T = (n as i32).into();
        let val3: T = TAU.into();
        let xd: Complex<T> = Complex::new(T::zero(), (val1 / val2) * val3).exp();

        res[t] = evens_res[t] + xd * odds_res[t];
        res[t + n / 2] = evens_res[t] - xd * odds_res[t];
    }

    return res;
}

fn fft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, -1);
    //.iter()
    //.map(|x| x / data.len() as f64)
    //.collect();
}

fn ifft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, 1);
}

fn main() {
    {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 44100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create("sine.wav", spec).unwrap();
        for t in (0..44100).map(|x| x as f64 / 44100.0) {
            let sample =
                0.5 * (t * 3400.0 * 5.0 / 4.0 * TAU).sin() + 0.7 * (t * 3400.0 * TAU).sin();
            let amplitude = i16::MAX as f64;
            writer.write_sample((sample * amplitude) as i16).unwrap();
        }
        for t in (0..44100).map(|x| x as f64 / 44100.0) {
            let sample =
                0.5 * (t * 1000.0 * 2.0 * TAU).sin() + 0.7 * (t * 1000.0 * TAU).sin();
            let amplitude = i16::MAX as f64;
            writer.write_sample((sample * amplitude) as i16).unwrap();
        }
    }

    let data: Vec<Complex64> = vec![1.0.into(), 10.0.into(), 100.0.into(), 1000.0.into()];

    println!("{:?}", data);
    let bruh = fft(&data);
    println!("{:?}", bruh);
    println!("{:?}", ifft(&bruh));

    //let mut reader = hound::WavReader::open("samples/120_G#_Leader_01_53_SP.wav").unwrap();
    let mut reader = hound::WavReader::open("sine.wav").unwrap();

    println!("{:?}", reader.spec());

    let channels = reader.spec().channels as usize;
    let bits = reader.spec().bits_per_sample;
    let max_val = (2 as u32).pow(bits as u32);

    let all_samples: Vec<f64> = reader
        .samples::<i32>()
        .step_by(channels)
        .map(|x| x.unwrap())
        .map(|x| x as f64 / max_val as f64)
        .collect();

    let (mut rl, thread) = raylib::init().size(2048, 480).title("Hello, World").build();

    let mut time: f64 = 0.0;
    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);
        time += d.get_frame_time() as f64;
        let curr_sample_start = (time * (reader.spec().sample_rate as f64)) as usize;
        println!("{}", curr_sample_start);

        if curr_sample_start + 2048 < all_samples.len() {
            let samples = &all_samples[curr_sample_start..(curr_sample_start + 2048)];

            let data: Vec<Complex64> = samples.iter().map(|x| c64(x.clone(), 0.0)).collect();
            let asd: Vec<Complex64> = fft(&data).iter().map(|x| x / data.len() as f64).collect();

            for i in 0..samples.len() {
                d.draw_line(
                    i as i32,
                    0,
                    i as i32,
                    (asd[i].abs() * 480.0 * 50.0) as i32,
                    Color::BLACK,
                );
            }
        }
    }
}
