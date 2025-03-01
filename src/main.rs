use std::f64::consts::TAU;

use raylib::prelude::*;

use num_complex::{c64, Complex64};

fn fft(ref data: Vec<Complex64>) -> Vec<Complex64> {
    let n = data.len();
    assert_eq!(n.count_ones(), 1);

    if n == 1 {
        return data.clone();
    }

    let evens: Vec<Complex64> = data.iter().step_by(2).cloned().collect();
    let odds: Vec<Complex64> = data.iter().skip(1).step_by(2).cloned().collect();

    let evens_res = fft(evens);
    let odds_res = fft(odds);

    let mut res: Vec<Complex64> = vec![c64(0, 0); n];
    for t in 0..(n / 2) {
        let xd = c64(0.0, -(t as f64) / (n as f64) * TAU).exp();
        res[t] = evens_res[t] + xd * odds_res[t];
        res[t + n / 2] = evens_res[t] - xd * odds_res[t];
    }

    return res;
}

fn main() {
    let (mut rl, thread) = raylib::init().size(640, 480).title("Hello, World").build();

    let data: Vec<Complex64> = vec![1.0.into(), 10.0.into(), 100.0.into(), 1000.0.into()];

    println!("{:?}", data);
    println!("{:?}", fft(data));
    println!("{:?}", c64(0.0, -(1 as f64) / (5 as f64) * TAU).exp());

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::WHITE);
        d.draw_text("Hello, world!", 12, 12, 20, Color::BLACK);
    }
}
