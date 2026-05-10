//! Shared fitting helpers used by overlap_and_chi4, self_intermediate_scattering,
//! polarity_tau, etc. Ported verbatim from legacy `observables.rs`.

/// Fit y(t) to exp(-(t/τ)^β) via Gauss-Newton with LM damping.
/// Returns (τ, β, R²).
pub fn fit_stretched_exponential(lag_times: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let data: Vec<(f64, f64)> = lag_times.iter().zip(y.iter())
        .filter(|(&t, &v)| t > 0.0 && v > 0.05 && v < 0.99)
        .map(|(&t, &v)| (t, v))
        .collect();
    if data.len() < 5 { return (f64::NAN, f64::NAN, 0.0); }

    let e_inv = 1.0 / std::f64::consts::E;
    let tau0 = data.iter()
        .min_by(|(_, a), (_, b)| (a - e_inv).abs().partial_cmp(&(b - e_inv).abs()).unwrap())
        .map_or(data[data.len() / 2].0, |&(t, _)| t)
        .max(1e-3);

    let mut tau = tau0;
    let mut beta = 0.8_f64;
    for _ in 0..100 {
        let mut jtj = [[0.0_f64; 2]; 2];
        let mut jtr = [0.0_f64; 2];
        for &(t, y_obs) in &data {
            let ratio = t / tau;
            let rb = ratio.powf(beta);
            let y_pred = (-rb).exp();
            let residual = y_obs - y_pred;
            let dy_dtau = y_pred * beta * rb / tau;
            let ln_ratio = ratio.ln();
            let dy_dbeta = -y_pred * rb * ln_ratio;
            jtj[0][0] += dy_dtau * dy_dtau;
            jtj[0][1] += dy_dtau * dy_dbeta;
            jtj[1][0] += dy_dbeta * dy_dtau;
            jtj[1][1] += dy_dbeta * dy_dbeta;
            jtr[0] += dy_dtau * residual;
            jtr[1] += dy_dbeta * residual;
        }
        let lambda = 1e-4;
        jtj[0][0] += lambda;
        jtj[1][1] += lambda;
        let det = jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0];
        if det.abs() < 1e-30 { break; }
        let d_tau = (jtj[1][1] * jtr[0] - jtj[0][1] * jtr[1]) / det;
        let d_beta = (jtj[0][0] * jtr[1] - jtj[1][0] * jtr[0]) / det;
        tau = (tau + d_tau).max(1e-3);
        beta = (beta + d_beta).clamp(0.01, 2.0);
        if (d_tau / tau).abs() < 1e-8 && (d_beta / beta).abs() < 1e-8 { break; }
    }
    let y_mean = data.iter().map(|&(_, y)| y).sum::<f64>() / data.len() as f64;
    let ss_tot: f64 = data.iter().map(|&(_, y)| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = data.iter()
        .map(|&(t, y)| (y - (-(t / tau).powf(beta)).exp()).powi(2)).sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    (tau, beta, r2)
}

/// Fit C(Δt) = exp(-Δt/τ) via linear regression on ln(C) vs Δt.
/// Returns (τ, R²). Returns (0, 0) if slope is non-decaying.
pub fn fit_exp_decay(lag_times: &[f64], corr: &[f64]) -> (f64, f64) {
    let n = lag_times.len() as f64;
    if n < 2.0 { return (0.0, 0.0); }
    let ln_corr: Vec<f64> = corr.iter().map(|c| c.ln()).collect();
    let sum_x: f64 = lag_times.iter().sum();
    let sum_y: f64 = ln_corr.iter().sum();
    let sum_xy: f64 = lag_times.iter().zip(ln_corr.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = lag_times.iter().map(|x| x * x).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 { return (0.0, 0.0); }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    if slope >= 0.0 { return (0.0, 0.0); }
    let tau = -1.0 / slope;
    let y_mean = sum_y / n;
    let ss_tot: f64 = ln_corr.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = lag_times.iter().zip(ln_corr.iter())
        .map(|(x, y)| { let yp = slope * x + intercept; (y - yp).powi(2) }).sum();
    let r2 = if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { 0.0 };
    (tau, r2)
}
