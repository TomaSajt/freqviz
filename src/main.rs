mod audio;
mod fft;

use std::{
    collections::VecDeque,
    path::Path,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

use audio::{make_microphone_input_audio_stream, make_output_audio_stream};
use cpal::traits::StreamTrait;

use eframe::egui::{self, pos2, vec2, Color32};

use fft::{fft, spinner_freqs_to_real_freqs};
use num_complex::Complex64;

fn main() -> Result<(), eframe::Error> {
    eframe::run_native(
        "FreqViz",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 700.0]),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::<MyApp>::new(MyApp::new(cc)))),
    )
}

#[derive(PartialEq)]
enum VizMode {
    Column,
    TopLine,
}

enum AudioStreamMode {
    Off,
    File(cpal::Stream, JoinHandle<()>),
    Microphone(cpal::Stream),
}

struct MyApp {
    audio_stream_mode: AudioStreamMode,

    kernel_size: usize,
    deque_mutex: Arc<Mutex<VecDeque<f64>>>,
    sample_rate: u32,
    viz_mode: VizMode,
    history_texture: Option<egui::TextureHandle>,
    scanline_prog: usize,
}

impl MyApp {
    fn new(_cc: &eframe::CreationContext) -> MyApp {
        MyApp {
            audio_stream_mode: AudioStreamMode::Off,
            kernel_size: 2048,
            deque_mutex: Arc::new(Mutex::new(VecDeque::<f64>::new())),
            sample_rate: 44100,
            viz_mode: VizMode::Column,
            history_texture: None,
            scanline_prog: 0,
        }
    }

    fn toggle_audio_stream_mode(&mut self) {
        self.deque_mutex.lock().unwrap().clear();
        match self.audio_stream_mode {
            AudioStreamMode::Off => {
                let (input_audio_stream, samples_per_sec) =
                    make_microphone_input_audio_stream(self.deque_mutex.clone()).unwrap();
                input_audio_stream.play().unwrap();
                self.audio_stream_mode = AudioStreamMode::Microphone(input_audio_stream);
                self.sample_rate = samples_per_sec;
            }
            AudioStreamMode::Microphone(_) => {
                let (output_audio_stream, join_handle, samples_per_sec) =
                    make_output_audio_stream(Path::new("sound.wav"), self.deque_mutex.clone())
                        .unwrap();
                output_audio_stream.play().unwrap();
                self.audio_stream_mode = AudioStreamMode::File(output_audio_stream, join_handle);
                self.sample_rate = samples_per_sec;
            }
            AudioStreamMode::File(_, _) => {
                self.audio_stream_mode = AudioStreamMode::Off;
                self.sample_rate = 44100;
            }
        }
    }

    fn draw_height_graph(&mut self, ui: &mut egui::Ui, real_freqs: &Vec<(f64, f64)>) {
        let main_line_col = if ui.visuals().dark_mode {
            Color32::from_additive_luminance(196)
        } else {
            Color32::from_black_alpha(240)
        };

        egui::ScrollArea::horizontal().show(ui, |ui| {
            egui::Frame::canvas(ui.style())
                .inner_margin(egui::Margin {
                    left: 30,
                    right: 50,
                    top: 0,
                    bottom: 25,
                })
                .show(ui, |ui| {
                    let desired_size =
                        vec2(ui.available_width() * 4.0, ui.available_height() * 0.5);
                    let (_id, rect) = ui.allocate_space(desired_size);

                    let to_screen = egui::emath::RectTransform::from_to(
                        egui::Rect::from_x_y_ranges(0.0..=1.0, 1.0..=0.0),
                        rect,
                    );

                    let num_freqs = real_freqs.len();

                    match self.viz_mode {
                        VizMode::Column => {
                            for i in 0..num_freqs {
                                let p = i as f32 / (num_freqs - 1) as f32;
                                ui.painter().line_segment(
                                    [
                                        to_screen * pos2(p, 0.0),
                                        to_screen * pos2(p, real_freqs[i].0 as f32),
                                    ],
                                    egui::Stroke::new(1.0, main_line_col),
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
                                .line(points, egui::epaint::PathStroke::new(1.0, main_line_col));
                        }
                    }

                    ui.painter().line_segment(
                        [to_screen * pos2(0.0, 0.0), to_screen * pos2(1.0, 0.0)],
                        egui::Stroke::new(1.0, main_line_col),
                    );

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
                        (self.sample_rate / 2) as f32,
                    ] {
                        let p = egui::remap(freq, 0.0..=((self.sample_rate / 2) as f32), 0.0..=1.0);

                        let text = freq.to_string() + "Hz";
                        let font_size = 20.0;

                        ui.painter().line_segment(
                            [to_screen * pos2(p, 0.0), to_screen * pos2(p, 1.0)],
                            egui::Stroke::new(1.0, Color32::LIGHT_GREEN),
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

                    ui.ctx().request_repaint();
                });
        });
    }

    fn draw_history_image(&mut self, ui: &mut egui::Ui, real_freqs: &Vec<(f64, f64)>) {
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
                egui::ColorImage::new([1000, real_freqs.len()], Color32::BLACK),
                egui::TextureOptions::NEAREST,
            ));
            self.scanline_prog = 0;
        }

        let data_line_image = egui::ColorImage {
            size: [1, real_freqs.len()],
            pixels: real_freqs
                .iter()
                .map(|a| {
                    let y = egui::remap_clamp(a.0, 0.0..=1.0, 0.0..=255.0) as u8;
                    Color32::from_rgb(y, y, 0)
                })
                .collect(),
        };

        let eraser_image = egui::ColorImage::new([1, real_freqs.len()], Color32::LIGHT_GREEN);

        if let Some(tex) = &mut self.history_texture {
            tex.set_partial(
                [self.scanline_prog, 0],
                data_line_image,
                egui::TextureOptions::NEAREST,
            );
            tex.set_partial(
                [(self.scanline_prog + 1) % 1000, 0],
                eraser_image,
                egui::TextureOptions::NEAREST,
            );
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
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
            ui.label(format!(
                "Current audio stream mode: {}",
                match self.audio_stream_mode {
                    AudioStreamMode::Off => "Off",
                    AudioStreamMode::File(_, _) => "File",
                    AudioStreamMode::Microphone(_) => "Microphone",
                }
            ));
            if ui.button("Toggle audio stream mode").clicked() {
                self.toggle_audio_stream_mode()
            }

            ui.separator();

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
                    .add_enabled(self.kernel_size > 1, egui::Button::new("x0.5"))
                    .clicked()
                {
                    self.kernel_size /= 2;
                }
            });

            ui.separator();

            let data = self.get_audio_buffer_data(self.kernel_size);
            let spinner_freqs = fft(&data);
            let real_freqs = spinner_freqs_to_real_freqs(spinner_freqs);

            ui.push_id("height_graph", |ui| {
                self.draw_height_graph(ui, &real_freqs);
            });

            ui.push_id("history", |ui| {
                self.draw_history_image(ui, &real_freqs);
            });
        });
    }
}
