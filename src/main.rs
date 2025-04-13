use std::{
    collections::VecDeque,
    f64::consts::TAU,
    sync::{Arc, Mutex},
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Stream,
};

use eframe::{
    egui::{
        self, pos2, remap_clamp, vec2, Button, Color32, ColorImage, Frame, Rect, ScrollArea,
        Stroke, TextureHandle, TextureOptions, Ui,
    },
    emath,
    epaint::PathStroke,
};

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

pub fn fft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, -1)
        .iter()
        .map(|x| x / data.len() as f64)
        .collect();
}

pub fn ifft(data: &Vec<Complex64>) -> Vec<Complex64> {
    return fft_helper(data, 1);
}

fn spinner_freqs_to_real_freqs(spinner_freqs: Vec<Complex64>) -> Vec<(f64, f64)> {
    let mut real_freqs: Vec<(f64, f64)> = vec![(0.0, 0.0); spinner_freqs.len() + 1];

    // we know that spinner_freqs[i]*exp(i/n*tau*t)+spinner_freqs[n-i]*exp(-i/n*tau*t) has to sum to a real number
    // this way we know that = spinner_freqs[i] is the complex conjugate of spinner_freqs[n-i]

    // we don't want negative amplitudes
    // if it happens, we just store a half-turn phaseshift instead

    real_freqs[0] = (spinner_freqs[0].abs(), spinner_freqs[0].arg());
    assert!(spinner_freqs[0].im().abs() < 0.001);

    real_freqs[spinner_freqs.len() / 2] = (
        spinner_freqs[spinner_freqs.len() / 2].abs(),
        spinner_freqs[spinner_freqs.len() / 2].arg(),
    );
    assert!(spinner_freqs[spinner_freqs.len() / 2].im().abs() < 0.001);

    for i in 1..(spinner_freqs.len() / 2) {
        // (a+bi)*exp(wit)+(a-bi)*exp(-wit) = 2a*cos(wt)-2b*cos(wt) = 2*sqrt(a^2+b^2)*cos(wt+atan2(b,a))
        real_freqs[i] = (2.0 * spinner_freqs[i].abs(), spinner_freqs[i].arg());
    }

    real_freqs
}

#[test]
fn test_fft() {
    fn is_almost_same(a: Complex64, b: Complex64) -> bool {
        (a - b).abs() < 0.00000001
    }

    let a: Vec<Complex64> = vec![1.0.into(), 10.0.into(), 100.0.into(), 1000.0.into()];
    let b = fft(&a);
    let c = ifft(&b);
    let d = a.into_iter().zip(c).all(|x| is_almost_same(x.0, x.1));
    assert!(d);
}

fn make_microphone_input_audio_stream(
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
) -> Result<(Stream, u32), anyhow::Error> {
    let host = cpal::default_host();

    // Set up the input device and stream with the default input config.
    let device = host
        .default_input_device()
        .expect("Failed to find input device");

    println!("Default input device: {}", device.name()?);

    let config = device
        .default_input_config()
        .expect("Failed to get default input config");

    let samples_per_sec = config.config().sample_rate.0;


    println!("sample rate: {}", samples_per_sec);

    println!("Default input config: {:?}", config);

    assert_eq!(config.config().channels, 2);

    let mut num_samples = 0;

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                let mut deque = deque_mutex.lock().unwrap();
                for sample in data.iter().step_by(2) {
                    deque.push_back(sample.clone() as f64);
                    num_samples += 1;
                    println!("{}", num_samples);
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

fn make_output_audio_stream(
    freq: f64,
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
) -> Result<(Stream, u32), anyhow::Error> {
    use cpal::traits::DeviceTrait;
    use cpal::traits::HostTrait;
    use cpal::SampleFormat;

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("Failed to find input device");

    println!("Default output device: {}", device.name()?);

    let supported_configs_ranges = device
        .supported_output_configs()
        .expect("error while querying configs")
        .collect::<Vec<_>>();

    println!("{:?}", supported_configs_ranges);

    let supported_config_range = supported_configs_ranges
        .iter()
        .find(|x| x.sample_format() == SampleFormat::F32 && x.channels() == 2)
        .expect("no f32 support found?");

    let supported_config = supported_config_range.with_sample_rate(cpal::SampleRate(44100));

    let sample_format = supported_config.sample_format();
    let sample_rate = supported_config.sample_rate().0;
    let channel_count = supported_config.channels();

    println!("sample rate: {}", sample_rate);

    let mut num_samples = 0;
    let mut curr_channel = 0;
    let stream = match sample_format {
        SampleFormat::F32 => device.build_output_stream(
            &supported_config.into(),
            move |data: &mut [f32], _: &_| {
                let mut deque = deque_mutex.lock().unwrap();
                println!("len: {}", data.len());
                for sample in data.as_mut() {
                    if curr_channel == 0 {
                        let t = num_samples as f64 / sample_rate as f64;
                        let val = (freq * TAU * t).sin();
                        *sample = val as f32;
                        deque.push_back(val);
                        num_samples += 1;
                        println!("{}", num_samples);
                    } else {
                        *sample = 0.0;
                    }
                    curr_channel = (curr_channel + 1) % channel_count;
                }
            },
            |err| eprintln!("an error occurred on the output audio stream: {}", err),
            None,
        )?,
        SampleFormat::I16 => {
            todo!()
        }
        SampleFormat::U16 => {
            todo!()
        }
        sample_format => panic!("Unsupported sample format '{sample_format}'"),
    };

    Ok((stream, sample_rate))
}

fn main() -> Result<(), eframe::Error> {
    eframe::run_native(
        "FreqViz",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 700.0]),
            ..Default::default()
        },
        Box::new(|_cc| Ok(Box::<MyApp>::new(MyApp::new().unwrap()))),
    )
}

#[derive(PartialEq)]
enum VizMode {
    Column,
    TopLine,
}

enum AudioStreamMode {
    Off,
    File(Stream),
    Microphone(Stream),
}

struct MyApp {
    audio_stream_mode: AudioStreamMode,

    kernel_size: usize,
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
    samples_per_sec: u32,
    viz_mode: VizMode,
    history_texture: Option<TextureHandle>,
    scanline_prog: usize,
}

impl MyApp {
    fn new() -> Result<Self, anyhow::Error> {
        let deque_mutex = Arc::new(Mutex::new(VecDeque::<f64>::new()));

        Ok(MyApp {
            audio_stream_mode: AudioStreamMode::Off,
            kernel_size: 2048,
            deque_mutex,
            samples_per_sec: 44100,
            viz_mode: VizMode::Column,
            history_texture: None,
            scanline_prog: 0,
        })
    }

    fn toggle_audio_stream_mode(&mut self) {
        self.deque_mutex.lock().unwrap().clear();
        match self.audio_stream_mode {
            AudioStreamMode::Off => {
                let (input_audio_stream, samples_per_sec) =
                    make_microphone_input_audio_stream(self.deque_mutex.clone()).unwrap();
                input_audio_stream.play().unwrap();
                self.audio_stream_mode = AudioStreamMode::Microphone(input_audio_stream);
                self.samples_per_sec = samples_per_sec;
            }
            AudioStreamMode::Microphone(_) => {
                let (output_audio_stream, samples_per_sec) =
                    make_output_audio_stream(440.0, self.deque_mutex.clone()).unwrap();
                output_audio_stream.play().unwrap();
                self.audio_stream_mode = AudioStreamMode::File(output_audio_stream);
                self.samples_per_sec = samples_per_sec;
            }
            AudioStreamMode::File(_) => {
                self.audio_stream_mode = AudioStreamMode::Off;
                self.samples_per_sec = 44100;
            }
        }
    }

    fn draw_height_graph(&mut self, ui: &mut Ui, real_freqs: &Vec<(f64, f64)>) {
        let main_line_col = if ui.visuals().dark_mode {
            Color32::from_additive_luminance(196)
        } else {
            Color32::from_black_alpha(240)
        };

        Frame::canvas(ui.style()).show(ui, |ui| {
            let desired_size = ui.available_width() * vec2(1.0, 0.35);
            let (_id, rect) = ui.allocate_space(desired_size);

            let to_screen =
                emath::RectTransform::from_to(Rect::from_x_y_ranges(0.0..=0.15, 1.0..=0.0), rect);

            let num_freqs = real_freqs.len();

            for freq in [
                0.0,
                110.0,
                110.0 * 2.0,
                110.0 * 4.0,
                110.0 * 8.0,
                110.0 * 16.0,
                110.0 * 32.0,
                110.0 * 64.0,
                110.0 * 128.0,
                (self.samples_per_sec / 2 + 1) as f32,
            ] {
                let p = emath::remap(
                    freq,
                    0.0..=((self.samples_per_sec / 2 + 1) as f32),
                    0.0..=1.0,
                );

                let text = freq.to_string() + "Hz";
                let font_size = 20.0;

                ui.painter().line_segment(
                    [to_screen * pos2(p, 0.0), to_screen * pos2(p, 1.0)],
                    Stroke::new(1.0, Color32::LIGHT_GREEN),
                );

                ui.painter().text(
                    to_screen * pos2(p, 0.0),
                    egui::Align2::CENTER_TOP,
                    text,
                    egui::FontId {
                        size: font_size,
                        family: egui::FontFamily::Monospace,
                    },
                    main_line_col,
                );
            }

            match self.viz_mode {
                VizMode::Column => {
                    for i in 0..num_freqs {
                        let p = i as f32 / (num_freqs - 1) as f32;
                        ui.painter().line_segment(
                            [
                                to_screen * pos2(p, 0.0),
                                to_screen * pos2(p, real_freqs[i].0 as f32),
                            ],
                            Stroke::new(1.0, main_line_col),
                        );
                    }
                }
                VizMode::TopLine => {
                    let points: Vec<_> = (0..num_freqs)
                        .map(|i| {
                            let p = i as f32 / (num_freqs - 1) as f32;
                            to_screen * pos2(p, real_freqs[i].0 as f32)
                        })
                        .collect();

                    ui.painter()
                        .line(points, PathStroke::new(1.0, main_line_col));
                }
            }

            ui.ctx().request_repaint();
        });

        ui.add_space(30.0);
    }

    fn draw_history_image(&mut self, ui: &mut Ui, real_freqs: &Vec<(f64, f64)>) {
        // Upload texture if it hasn't been already or if the data width changed
        if self
            .history_texture
            .as_ref()
            .map(|tex| tex.size()[1])
            .unwrap_or(0)
            != real_freqs.len()
        {
            self.history_texture = Some(ui.ctx().load_texture(
                "history_texture",
                ColorImage::new([1000, real_freqs.len()], Color32::BLACK),
                TextureOptions::NEAREST,
            ));
            self.scanline_prog = 0;
        }

        let data_line_image = ColorImage {
            size: [1, real_freqs.len()],
            pixels: real_freqs
                .iter()
                .map(|a| {
                    let y = remap_clamp(a.0, 0.0..=1.0, 0.0..=255.0) as u8;
                    Color32::from_rgb(y, y, 0)
                })
                .collect(),
        };

        let eraser_image = ColorImage::new([1, real_freqs.len()], Color32::LIGHT_GREEN);

        if let Some(tex) = &mut self.history_texture {
            tex.set_partial(
                [self.scanline_prog, 0],
                data_line_image,
                TextureOptions::NEAREST,
            );
            tex.set_partial(
                [(self.scanline_prog + 1) % 1000, 0],
                eraser_image,
                TextureOptions::NEAREST,
            );
        }

        ScrollArea::vertical().show(ui, |ui| {
            ui.image(self.history_texture.as_ref().unwrap());
        });

        self.scanline_prog = (self.scanline_prog + 1) % 1000;
    }

    fn get_audio_buffer_data(&mut self, len: usize) -> Vec<Complex64> {
        let mut deque = self.deque_mutex.lock().unwrap();
        // only keep as much data in the deque as we need
        while deque.len() > len {
            deque.pop_front();
        }
        // pad the data with zeroes if not enough data
        while deque.len() < len {
            deque.push_front(0.0);
        }
        deque
            .iter()
            .map(|x| Into::<Complex64>::into(x.clone()))
            .collect::<Vec<_>>()
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if ui.button("Toggle audio stream mode").clicked() {
                self.toggle_audio_stream_mode()
            }

            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.viz_mode, VizMode::Column, "Column");
                ui.selectable_value(&mut self.viz_mode, VizMode::TopLine, "Top Line");
            });

            ui.label(format!("Kernel size: {}", self.kernel_size));
            ui.horizontal(|ui| {
                if ui.button("x2").clicked() {
                    self.kernel_size *= 2;
                }
                if ui
                    .add_enabled(self.kernel_size > 1, Button::new("x0.5"))
                    .clicked()
                {
                    self.kernel_size /= 2;
                }
            });

            let data = self.get_audio_buffer_data(self.kernel_size);
            let spinner_freqs = fft(&data);
            let real_freqs = spinner_freqs_to_real_freqs(spinner_freqs);

            self.draw_height_graph(ui, &real_freqs);

            self.draw_history_image(ui, &real_freqs);
        });
    }
}
