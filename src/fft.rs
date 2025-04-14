use std::f64::consts::TAU;

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

pub fn spinner_freqs_to_real_freqs(spinner_freqs: Vec<Complex64>) -> Vec<(f64, f64)> {
    let mut real_freqs: Vec<(f64, f64)> = vec![(0.0, 0.0); spinner_freqs.len() / 2 + 1];

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
