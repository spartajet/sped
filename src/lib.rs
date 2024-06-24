use std::error::Error;
use std::f64::EPSILON;
use opencv::core::Point2d;
use opencv::prelude::{Mat, MatTraitConst};

pub fn devernay(image: &Vec<u8>, rows: usize, cols: usize, sigma: f64, th_h: f64, th_l: f64) -> Result<DevernayResult, Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut gx = vec![0f64; pixel_num as usize];
    let mut gy = vec![0f64; pixel_num as usize];
    let mut mod_g = vec![0f64; pixel_num as usize];
    if sigma == 0f64 {
        compute_gradient(image, rows, cols, &mut gx, &mut gy, &mut mod_g);
    } else {
        let gauss = gaussian_filter(image, rows, cols, sigma)?;
        compute_gradient(&gauss, rows, cols, &mut gx, &mut gy, &mut mod_g);
    }
    let (edge_x, edge_y) = compute_edge_points(&gx, &gy, &mod_g, rows, cols)?;
    let (mut next, mut prev) = chain_edge_points(&edge_x, &edge_y, &gx, &gy, rows, cols)?;
    thresholds_with_hysteresis(&mut next, &mut prev, &mod_g, rows, cols, th_h, th_l)?;

    let mut result = DevernayResult::default();
    Ok(result)
}

fn list_chained_edge_points(next: &mut Vec<i32>, prev: &mut Vec<i32>, edge_x: &Vec<f64>, edge_y: &Vec<f64>, rows: usize, cols: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<i32>, i32), Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut xs = vec![0f64; pixel_num];
    let mut ys = vec![0f64; pixel_num];
    let mut curve_limits = vec![0i32; pixel_num];
    // let mut i = 0;
    let mut k = 0;
    let mut n = 0i32;
    let mut nn = 0;
    let mut m = 0;
    for i in 0..pixel_num {
        if prev[i] >= 0 || next[i] >= 0 {
            curve_limits[m] = n;

            m = m + 1;
            k = i;
            n = prev[k];
            while n >= 0 && n != i as i32 {
                k = n as usize;
                n = prev[k];
            }
            while k >= 0 {
                xs[nn] = edge_x[k];
                ys[nn] = edge_y[k];
                nn = nn + 1;
                n = next[k];
                next[k] = -1;
                prev[k] = -1;
                k = n as usize;
            }
        }
        curve_limits[m] = nn as i32;
    }

    Ok((xs, ys, curve_limits, m as i32))
}

fn thresholds_with_hysteresis(next: &mut Vec<i32>, prev: &mut Vec<i32>, mod_g: &Vec<f64>, cols: usize, rows: usize, th_h: f64, th_l: f64) -> Result<(), Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut valid = vec![0; pixel_num];
    let mut j = 0;
    let mut k = 0;
    for i in 0..pixel_num {
        if (prev[i] >= 0 || next[i] >= 0) && valid[i] == 0 && mod_g[i] >= th_h {
            valid[i] = 1;
            j = i;
            k = next[j];
            while j >= 0 && k >= 0 && valid[k as usize] == 0 {
                if mod_g[k as usize] < th_l {
                    next[j] = -1;
                    prev[k as usize] = -1;
                } else {
                    valid[k as usize] = 1;
                }
                j = next[j] as usize;
            }
            j = i;
            k = prev[j];
            while j >= 0 && k >= 0 && valid[k as usize] == 0 {
                if mod_g[k as usize] < th_l {
                    next[k as usize] = -1;
                    prev[j] = -1;
                } else {
                    valid[k as usize] = 1;
                }
                j = prev[j] as usize;
            }
        }
    }
    for i in 0..pixel_num {
        if (prev[i] >= 0 || next[i] >= 0) && valid[i] == 0 {
            next[k as usize] = -1;
            prev[j] = -1;
        }
    }
    Ok(())
}

fn compute_gradient(image: &Vec<u8>, rows: usize, cols: usize, gx: &mut Vec<f64>, gy: &mut Vec<f64>, mod_g: &mut Vec<f64>) {
    // let mut i = 0;
    for i in 1..cols {
        for j in 1..rows {
            gx[i + j * cols] = (image[(i + 1) + j * cols] - image[(i - 1) + j * cols]) as f64;
            gy[i + j * cols] = (image[i + (j + 1) * cols] - image[i + (j - 1) * cols]) as f64;
            mod_g[i + j * cols] = (gx[i + j * cols].powi(2) + gy[i + j * cols].powi(2)).sqrt();
        }
    }
}

fn gaussian_filter(image: &Vec<u8>, rows: usize, cols: usize, sigma: f64) -> Result<Vec<u8>, Box<dyn Error>> {
    if sigma < 0f64 {
        return Err("Sigma must be positive".into());
    }
    if rows < 1 || cols < 1 {
        return Err("Rows and cols must be positive".into());
    }
    let len = (rows * cols);
    let mut out = vec![0; len];
    let mut tmp = vec![0f64; len];
    let prec = 3.0f64;
    let offset = (sigma * (2f64 * prec * 10f64.log10()).sqrt()).ceil() as usize;
    let n = 2 * offset + 1;
    let kernel = gaussian_kernel(n, sigma, offset as f64)?;
    let nx2 = 2 * cols;
    let ny2 = 2 * rows;
    // x axis convolution
    for x in 0..cols {
        for y in 0..rows {
            let mut value = 0f64;
            for i in 0..n {
                let mut j = x - offset + i;
                while j < 0 {
                    j = nx2 + j;
                }
                while j > nx2 {
                    j = j - nx2;
                }
                if j > cols {
                    j = nx2 - j - 1;
                }
                value += image[j + y * cols] as f64 * kernel[i];
            }
            tmp[x + y * cols] = value;
        }
    }

    // y axis convolution
    for x in 0..cols {
        for y in 0..rows {
            let mut value = 0f64;
            for i in 0..n {
                let mut j = y - offset + i;
                while j < 0 {
                    j = ny2 + j;
                }
                while j > ny2 {
                    j = j - ny2;
                }
                if j > rows {
                    j = ny2 - j - 1;
                }
                value += tmp[x + j * cols] as f64 * kernel[i];
            }
            out[x + y * cols] = value as u8;
        }
    }
    Ok(out)
}

fn gaussian_kernel(n: usize, sigma: f64, mean: f64) -> Result<Vec<f64>, Box<dyn Error>> {
    if sigma < 0f64 {
        return Err("Sigma must be positive".into());
    }
    if n < 1 {
        return Err("n must be positive".into());
    }
    if n % 2 == 0 {
        return Err("n must be odd".into());
    }
    let mut sum = 0f64;
    let mut kernel = vec![0f64; n as usize];
    for i in 0..n {
        let val = (i as f64 - mean) / sigma;
        let tmp = (-0.5 * val.powi(2)).exp();
        sum += tmp;
        kernel[i as usize] = tmp;
    }
    if sum > 0f64 {
        for i in 0..n {
            kernel[i as usize] /= sum;
        }
    }

    Ok(kernel)
}

fn compute_edge_points(gx: &Vec<f64>, gy: &Vec<f64>, mod_g: &Vec<f64>, rows: usize, cols: usize) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut edge_x = vec![-1f64; pixel_num as usize];
    let mut edge_y = vec![-1f64; pixel_num as usize];
    for x in 2..cols - 2 {
        for y in 2..rows - 2 {
            let mut dx = 0;
            let mut dy = 0;
            let mod_value = mod_g[x + y * cols];
            let mod_l = mod_g[x - 1 + y * cols];
            let mod_r = mod_g[x + 1 + y * cols];
            let mod_u = mod_g[x + (y + 1) * cols];
            let mod_d = mod_g[x + (y - 1) * cols];
            let gx = gx[x + y * cols].abs();
            let gy = gy[x + y * cols].abs();
            if greater_round(mod_value, mod_l) && !greater_round(mod_r, mod_value) && gx >= gy {
                dx = 1;
            } else if greater_round(mod_value, mod_d) && !greater_round(mod_u, mod_value) && gx <= gy {
                dy = 1;
            }
            if dx == 1 || dy == 1 {
                let a = mod_g[x - dx + (y - dy) * cols];
                let b = mod_g[x + y * cols];
                let c = mod_g[x + dx + (y + dy) * cols];
                let offset = 0.5 * (a - c) / (a - b - b + c);

                edge_x[x + y * cols] = x as f64 + offset * dx as f64;
                edge_y[x + y * cols] = y as f64 + offset * dy as f64;
            }
        }
    }
    Ok((edge_x, edge_y))
}

fn chain_edge_points(edge_x: &Vec<f64>, edge_y: &Vec<f64>, gx: &Vec<f64>, gy: &Vec<f64>, rows: usize, cols: usize) -> Result<(Vec<i32>, Vec<i32>), Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut next = vec![-1i32; pixel_num];
    let mut prev = vec![-1i32; pixel_num];
    let mut alt = 0i32;
    for x in 2..cols - 2 {
        for y in 2..rows - 2 {
            if edge_x[x + y * cols] < 0f64 || edge_y[x + y * cols] < 0f64 {
                continue;
            }
            let from = x + y * cols;
            let mut fwd_s = 0f64;
            let mut bck_s = 0f64;
            let mut fwd = -1i32;
            let mut bck = -1i32;
            for i in (-2..3).collect::<Vec<i32>>() {
                for j in (-2..3).collect::<Vec<i32>>() {
                    let to = (x + i as usize) + (y + j as usize) * cols;
                    let s = chain(from, to, edge_x, edge_y, gx, gy, cols, rows)?;
                    if s > fwd_s {
                        fwd_s = s;
                        fwd = to as i32;
                    }
                    if s < bck_s {
                        bck_s = s;
                        bck = to as i32;
                    }
                }
            }
            alt = prev[fwd as usize];
            let fwd_tmp = chain(alt as usize, fwd as usize, edge_x, edge_y, gx, gy, cols, rows)?;
            if (fwd >= 0 && next[from] != fwd as i32 && (alt < 0 || fwd_tmp < fwd_s))
            {
                if (next[from] >= 0) {
                    /* remove previous from-x link if one */
                    prev[next[from] as usize] = -1;
                } /* only prev requires explicit reset  */
                next[from] = fwd as i32;         /* set next of from-fwd link          */
                if (alt >= 0) {
                    /* remove alt-fwd link if one         */
                    next[alt as usize] = -1;
                }   /* only next requires explicit reset  */
                prev[fwd as usize] = from as i32;         /* set prev of from-fwd link          */
            }
            alt = next[bck as usize];
            let bck_tmp = chain(alt as usize, bck as usize, edge_x, edge_y, gx, gy, cols, rows)?;
            if (bck >= 0 && prev[from] != bck as i32 && (alt < 0 || bck_tmp > bck_s))
            {
                if (alt >= 0) {
                    /* remove bck-alt link if one         */
                    prev[alt as usize] = -1;
                }      /* only prev requires explicit reset  */
                next[bck as usize] = from as i32;         /* set next of bck-from link          */
                if (prev[from] >= 0) {
                    /* remove previous x-from link if one */
                    next[prev[from] as usize] = -1;
                } /* only next requires explicit reset  */
                prev[from] = bck as i32;         /* set prev of bck-from link          */
            }
        }
    }
    Ok((next, prev))
}

fn chain(from: usize, to: usize, edge_x: &Vec<f64>, edge_y: &Vec<f64>, gx: &Vec<f64>, gy: &Vec<f64>, cols: usize, rows: usize) -> Result<f64, Box<dyn Error>> {
    let mut dx = 0f64;
    let mut dy = 0f64;
    if from == to {
        return Ok(0f64);
    }
    if edge_x[from] < 0f64 || edge_y[from] < 0f64 || edge_x[to] < 0f64 || edge_y[to] < 0f64 {
        return Ok(0f64);
    }
    dx = edge_x[to] - edge_x[from];
    dy = edge_y[to] - edge_y[from];
    if (gy[from] * dx - gx[from] * dy) * (gy[to] * dx - gx[to] * dy) < 0f64 {
        return Ok(0f64);
    }
    Ok(0f64)
}

fn greater_round(a: f64, b: f64) -> bool {
    if a <= b {
        return false;
    }
    if a - b < f64::EPSILON {
        return false;
    }
    true
}


#[derive(Debug, Clone, Default)]
struct DevernayResult {
    xs: Vec<f64>,
    ys: Vec<f64>,
    edge_points_num: i32,
    curves_num: i32,
    curve_limits: i32,
}