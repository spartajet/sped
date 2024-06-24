use std::error::Error;
use opencv::core::{Size, Vector};
use opencv::imgcodecs::imwrite;
use opencv::prelude::Mat;

pub fn devernay(image: &Vec<u8>, rows: usize, cols: usize, sigma: f64, th_h: f64, th_l: f64) -> Result<DevernayResult, Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut gx = vec![0f64; pixel_num];
    let mut gy = vec![0f64; pixel_num];
    let mut mod_g = vec![0f64; pixel_num];
    if sigma == 0f64 {
        compute_gradient(image, rows, cols, &mut gx, &mut gy, &mut mod_g);
    } else {
        let gauss = gaussian_filter(image, rows, cols, sigma)?;
        let gauss_array = gauss.as_slice();
        let gauss_mat = Mat::new_size_with_data(Size::new(cols as i32, rows as i32), gauss_array).unwrap();
        imwrite("resources/gauss.png", &gauss_mat, &Vector::<i32>::new()).unwrap();
        for (i, v) in gauss.iter().enumerate() {
            if (*v < 255) {
                // println!("i: {}, v: {}", i, v);
            }
        }
        compute_gradient(&gauss, rows, cols, &mut gx, &mut gy, &mut mod_g);
    }
    for i in 0..pixel_num {
        if gx[i] == 0f64 || gy[i] == 0f64 || mod_g[i] == 0f64 { continue; }
        // println!("i:{} gx: {:.4}, gy: {:.4}, mod_g: {:.4}", i, gx[i], gy[i], mod_g[i]);
    }
    let (edge_x, edge_y) = compute_edge_points(&gx, &gy, &mod_g, rows, cols)?;
    let (mut next, mut prev) = chain_edge_points(&edge_x, &edge_y, &gx, &gy, rows, cols)?;
    thresholds_with_hysteresis(&mut next, &mut prev, &mod_g, rows, cols, th_h, th_l)?;
    let (xs, ys, curve_limits, curves_num) = list_chained_edge_points(&mut next, &mut prev, &edge_x, &edge_y, rows, cols)?;
    let result = DevernayResult {
        xs,
        ys,
        curve_limits,
        curves_num,
    };
    Ok(result)
}

fn list_chained_edge_points(next: &mut Vec<i32>, prev: &mut Vec<i32>, edge_x: &Vec<f64>, edge_y: &Vec<f64>, rows: usize, cols: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<i32>, i32), Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut xs = vec![0f64; pixel_num];
    let mut ys = vec![0f64; pixel_num];
    let mut curve_limits = vec![0i32; pixel_num];
    // let mut i = 0;
    // let mut k = 0;
    let mut n = 0i32;
    let mut nn = 0;
    let mut m = 0;
    for i in 0..pixel_num {
        if prev[i] >= 0 || next[i] >= 0 {
            curve_limits[m] = n;

            m = m + 1;
            let mut k = i as i32;
            n = prev[k as usize];
            while n >= 0 && n != i as i32 {
                k = n;
                n = prev[k as usize];
            }
            while k >= 0 {
                xs[nn] = edge_x[k as usize];
                ys[nn] = edge_y[k as usize];
                nn = nn + 1;
                n = next[k as usize];
                next[k as usize] = -1;
                prev[k as usize] = -1;
                k = n;
            }
        }
        curve_limits[m] = nn as i32;
    }

    Ok((xs, ys, curve_limits, m as i32))
}

fn thresholds_with_hysteresis(next: &mut Vec<i32>, prev: &mut Vec<i32>, mod_g: &Vec<f64>, cols: usize, rows: usize, th_h: f64, th_l: f64) -> Result<(), Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut valid = vec![0; pixel_num];
    let mut j = 0i32;
    let mut k = 0;
    for i in 0..pixel_num {
        if (prev[i] >= 0 || next[i] >= 0) && valid[i] == 0 && mod_g[i] >= th_h {
            valid[i] = 1;
            j = i as i32;
            k = next[j as usize];
            while j >= 0 && k >= 0 && valid[k as usize] == 0 {
                if mod_g[k as usize] < th_l {
                    next[j as usize] = -1;
                    prev[k as usize] = -1;
                } else {
                    valid[k as usize] = 1;
                }
                j = next[j as usize];
            }
            j = i as i32;
            k = prev[j as usize];
            while j >= 0 && k >= 0 && valid[k as usize] == 0 {
                if mod_g[k as usize] < th_l {
                    next[k as usize] = -1;
                    prev[j as usize] = -1;
                } else {
                    valid[k as usize] = 1;
                }
                j = prev[j as usize];
            }
        }
    }
    for i in 0..pixel_num {
        if (prev[i] >= 0 || next[i] >= 0) && valid[i] == 0 {
            next[i] = -1;
            prev[i] = -1;
        }
    }
    Ok(())
}

fn compute_gradient(image: &Vec<u8>, rows: usize, cols: usize, gx: &mut Vec<f64>, gy: &mut Vec<f64>, mod_g: &mut Vec<f64>) {
    // let mut i = 0;
    for i in 1..cols - 1 {
        for j in 1..rows - 1 {
            gx[i + j * cols] = image[(i + 1) + j * cols] as f64 - image[(i - 1) + j * cols] as f64;
            gy[i + j * cols] = image[i + (j + 1) * cols] as f64 - image[i + (j - 1) * cols] as f64;
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
    let len = rows * cols;
    let mut out = vec![0; len];
    let mut tmp = vec![0f64; len];
    let prec = 3.0f64;
    let offset = (sigma * (2f64 * prec * 10f64.ln()).sqrt()).ceil() as i32;
    let n = 2 * offset + 1;
    let kernel = gaussian_kernel(n as usize, sigma, offset as f64)?;
    let nx2 = 2 * cols;
    let ny2 = 2 * rows;
    // x axis convolution
    for x in 0..cols {
        for y in 0..rows {
            let mut value = 0f64;
            for i in 0..n {
                let mut j = x as i32 - offset + i;
                while j < 0 {
                    j = nx2 as i32 + j;
                }
                while j > nx2 as i32 {
                    j = j - nx2 as i32;
                }
                if j >= cols as i32 {
                    j = nx2 as i32 - j - 1;
                }
                value += image[j as usize + y * cols] as f64 * kernel[i as usize];
            }
            tmp[x + y * cols] = value;
        }
    }

    // y axis convolution
    for x in 0..cols {
        for y in 0..rows {
            let mut value = 0f64;
            for i in 0..n {
                let mut j = y as i32 - offset + i;
                while j < 0 {
                    j = ny2 as i32 + j;
                }
                while j > ny2 as i32 {
                    j = j - ny2 as i32;
                }
                if j >= rows as i32 {
                    j = ny2 as i32 - j - 1;
                }
                value += tmp[x + j as usize * cols] * kernel[i as usize];
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
    let mut kernel = vec![0f64; n];
    for i in 0..n {
        let val = (i as f64 - mean) / sigma;
        let tmp = (-0.5 * val.powi(2)).exp();
        sum += tmp;
        kernel[i] = tmp;
    }
    if sum > 0f64 {
        for i in 0..n {
            kernel[i] /= sum;
        }
    }

    Ok(kernel)
}

fn compute_edge_points(gx: &Vec<f64>, gy: &Vec<f64>, mod_g: &Vec<f64>, rows: usize, cols: usize) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let pixel_num = rows * cols;
    let mut edge_x = vec![-1f64; pixel_num];
    let mut edge_y = vec![-1f64; pixel_num];
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
            if mod_value > mod_l && mod_value > mod_r && gx >= gy {
                dx = 1;
            } else if mod_value > mod_d && mod_value > mod_u && gx <= gy {
                dy = 1;
            }
            if dx > 0 || dy > 0 {
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
    // let mut alt = 0i32;
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
                    let to = (x as i32 + i) + (y as i32 + j) * cols as i32;
                    let s = chain(from, to as usize, edge_x, edge_y, gx, gy, cols, rows)?;
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
            let mut alt = 0;
            if fwd > 0 {
                if next[from] != fwd {
                    alt = prev[fwd as usize];
                    let mut cal = false;
                    if alt < 0 {
                        cal = true;
                    }else {
                        let fwd_tmp = chain(alt as usize, fwd as usize, edge_x, edge_y, gx, gy, cols, rows)?;
                        if fwd_tmp < fwd_s {
                            cal = true;
                        }
                    }
                    if cal {
                        if next[from] >= 0 {
                            /* remove previous from-x link if one */
                            prev[next[from] as usize] = -1;
                        } /* only prev requires explicit reset  */
                        next[from] = fwd;         /* set next of from-fwd link          */
                        if alt >= 0 {
                            /* remove alt-fwd link if one         */
                            next[alt as usize] = -1;
                        }   /* only next requires explicit reset  */
                        prev[fwd as usize] = from as i32;         /* set prev of from-fwd link          */
                    }
                }
            }

            if bck > 0 {
                if prev[from] != bck {
                    alt = next[bck as usize];
                    let mut cal = false;
                    if alt < 0 {
                        cal = true;
                    } else {
                        let bck_tmp = chain(alt as usize, bck as usize, edge_x, edge_y, gx, gy, cols, rows)?;
                        if bck_tmp > bck_s {
                            cal = true;
                        }
                    }
                    if cal{
                        if alt >= 0 {
                            /* remove bck-alt link if one         */
                            prev[alt as usize] = -1;
                        }      /* only prev requires explicit reset  */
                        next[bck as usize] = from as i32;         /* set next of bck-from link          */
                        if prev[from] >= 0 {
                            /* remove previous x-from link if one */
                            next[prev[from] as usize] = -1;
                        } /* only next requires explicit reset  */
                        prev[from] = bck;         /* set prev of bck-from link          */
                    }
                }
            }
        }
    }
    Ok((next, prev))
}

fn chain(from: usize, to: usize, edge_x: &Vec<f64>, edge_y: &Vec<f64>, gx: &Vec<f64>, gy: &Vec<f64>, _cols: usize, _rows: usize) -> Result<f64, Box<dyn Error>> {
    // let mut dx = 0f64;
    // let mut dy = 0f64;
    if from == to {
        return Ok(0f64);
    }
    if edge_x[from] < 0f64 || edge_y[from] < 0f64 || edge_x[to] < 0f64 || edge_y[to] < 0f64 {
        return Ok(0f64);
    }
    let dx = edge_x[to] - edge_x[from];
    let dy = edge_y[to] - edge_y[from];
    if (gy[from] * dx - gx[from] * dy) * (gy[to] * dx - gx[to] * dy) < 0f64 {
        return Ok(0f64);
    }
    let mut result = 0f64;
    if gy[from] * dx - gx[from] * dy >= 0f64 {
        result = 1f64 / dist(edge_x[from], edge_y[from], edge_x[to], edge_y[to]);
    } else {
        result = -1f64 / dist(edge_x[from], edge_y[from], edge_x[to], edge_y[to]);
    }
    Ok(result)
}

fn dist(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x1 - x2).powi(2) + (y1 - y2).powi(2)).sqrt()
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
pub struct DevernayResult {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    // pub edge_points_num: i32,
    pub curves_num: i32,
    pub curve_limits: Vec<i32>,
}