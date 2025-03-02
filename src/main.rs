use std::f64::consts::TAU;

use raylib::prelude::*;

use num_complex::{Complex64, ComplexFloat};

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

fn fft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, -1)
        .iter()
        .map(|x| x / data.len() as f64)
        .collect();
}

fn ifft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, 1);
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
        let sample = 0.3 * (t * 261.6256 / 44100.0 * TAU).sin();

        if sample > 1.0 {
            panic!("Sample is cutting off!");
        }
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..44100).map(|x| x as f64) {
        let sample = 0.3 * (t * 329.6276 / 44100.0 * TAU).sin();

        if sample > 1.0 {
            panic!("Sample is cutting off!");
        }
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..44100).map(|x| x as f64) {
        let sample = 0.3 * (t * 391.9954 / 44100.0 * TAU).sin();

        if sample > 1.0 {
            panic!("Sample is cutting off!");
        }
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
    for t in (0..(44100 * 3)).map(|x| x as f64) {
        let sample = 0.3 * (t * 261.6256 / 44100.0 * TAU).sin()
            + 0.3 * (t * 329.6276 / 44100.0 * TAU).sin()
            + 0.3 * (t * 391.9954 / 44100.0 * TAU).sin();

        if sample > 1.0 {
            panic!("Sample is cutting off!");
        }
        writer.write_sample((sample * amplitude) as i16).unwrap();
    }
}

fn main() {
    write_sample();

    {
        let data: Vec<Complex64> = vec![1.0.into(), 10.0.into(), 100.0.into(), 1000.0.into()];

        println!("{:?}", data);
        let bruh = fft(&data);
        println!("{:?}", bruh);
        println!("{:?}", ifft(&bruh));
    }

    //let mut reader = hound::WavReader::open("samples/120_G#_Leader_01_53_SP.wav").unwrap();
    let mut reader = hound::WavReader::open("sine.wav").unwrap();

    println!("{:?}", reader.spec());

    let channels = reader.spec().channels as usize;
    let bits = reader.spec().bits_per_sample;
    let max_val: i32 = 1 << (bits - 1);

    let all_int_samples: Vec<_> = reader
        .samples::<i32>()
        .step_by(channels)
        .map(|x| x.unwrap())
        .collect();

    let all_samples: Vec<f64> = all_int_samples
        .iter()
        .map(|x| x.clone() as f64 / max_val as f64)
        .map(|x| if x > 1.0 { panic!("nooo") } else { x })
        .collect();

    let asd: i32 = all_int_samples.into_iter().max().unwrap();
    println!("max_val: {}", max_val);
    println!("Asd: {}", asd);

    let (mut rl, thread) = raylib::init().size(2048, 480).title("Hello, World").build();

    let mut kernel_size: usize = 2048;
    let mut time: f64 = 0.0;
    while !rl.window_should_close() {
        if rl.is_key_pressed(KeyboardKey::KEY_UP) {
            kernel_size *= 2;
        }
        if kernel_size > 1 && rl.is_key_pressed(KeyboardKey::KEY_DOWN) {
            kernel_size /= 2;
        }

        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::WHITE);
        time += d.get_frame_time() as f64;
        let curr_sample_start = (time * (reader.spec().sample_rate as f64)) as usize;
        println!("time={}, sample_start={}", time, curr_sample_start);

        let w = d.get_screen_width();
        let h = d.get_screen_height();

        if curr_sample_start + kernel_size < all_samples.len() {
            let samples = &all_samples[curr_sample_start..(curr_sample_start + kernel_size)];

            let data: Vec<Complex64> = samples
                .iter()
                .map(|x| Into::<Complex64>::into(x.clone()))
                .collect();
            let res: Vec<Complex64> = fft(&data);

            for x in 0..w {
                let prog = x as f32 / w as f32;

                let rot_per_sec = lerp(0.0, 1200.0, prog);
                let rot_per_sample = rot_per_sec / 44100.0;
                let fract_rot_per_sample = rot_per_sample * kernel_size as f32;
                let i = fract_rot_per_sample as i32;
                if i < 0 {
                    continue;
                }
                let i = i as usize;
                if i >= kernel_size {
                    continue;
                }

                d.draw_line(
                    x as i32,
                    0,
                    x as i32,
                    (res[i].abs() * 3.0 * h as f64) as i32,
                    Color::BLACK,
                );
            }
        }
    }
}
