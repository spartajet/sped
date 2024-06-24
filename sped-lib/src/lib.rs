use std::f64::consts::E;

// fn gaussian_kernel(pf_gauss_filter: &mut [f64], n: usize, sigma: f64, mean: f64) {
//     let mut sum = 0.0;
//     let mut val: f64;
//     if pf_gauss_filter.is_empty() {
//         panic!("gaussian_kernel: kernel not allocated");
//     }
//     if sigma <= 0.0 {
//         panic!("gaussian_kernel: sigma must be positive");
//     }
//     // compute Gaussian kernel
//     for i in 0..n {
//         val = ((i as f64) - mean) / sigma;
//         pf_gauss_filter[i] = E.powf(-0.5 * val * val);
//         sum += pf_gauss_filter[i];
//     }
//     // normalization
//     if sum > 0.0 {
//         for i in 0..n {
//             pf_gauss_filter[i] /= sum;
//         }
//     }
// }

fn greater_round(a: f64, b: f64) -> bool {
    const EPSILON: f64 = f64::EPSILON;
    const THRESHOLD: f64 = 1000.0 * EPSILON;
    if a <= b {
        return false;
    }
    if (a - b) < THRESHOLD {
        return false;
    }

    true
}

fn gaussian_kernel(n: usize, sigma: f64, mean: f64) -> Vec<f64> {
    let mut kernel = Vec::with_capacity(n);
    for i in 0..n {
        let val = ((i as f64) - mean) / sigma;
        kernel.push(E.powf(-0.5 * val * val));
    }
    let sum: f64 = kernel.iter().sum();
    kernel.iter().map(|x| x / sum).collect()
}

fn dist(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).sqrt()
}

fn gaussian_filter(image: &Vec<u8>, out: &mut Vec<u8>, X: usize, Y: usize, sigma: f64) {
    let mut tmp = vec![0.0; X * Y];
    let prec = 3.0;
    let offset = (sigma * (2.0 * prec * 10f64.ln()).sqrt()).ceil() as usize;
    let n = 1 + 2 * offset;
    let kernel = gaussian_kernel(n, sigma, offset as f64);
    let nx2 = 2 * X;
    let ny2 = 2 * Y;
    for x in 0..X {
        for y in 0..Y {
            let mut val = 0.0;
            for i in 0..n {
                let mut j = x as i32 - offset as i32 + i as i32;
                while j < 0 {
                    j += nx2 as i32;
                }
                while j >= nx2 as i32 {
                    j -= nx2 as i32;
                }
                if j >= X as i32 {
                    j = nx2 as i32 - 1 - j;
                }
                val += (image[j as usize + y * X] as f64) * kernel[i];
            }
            tmp[x + y * X] = val;
        }
    }
    for x in 0..X {
        for y in 0..Y {
            let mut val = 0.0;
            for i in 0..n {
                let mut j = y as i32 - offset as i32 + i as i32;
                while j < 0 {
                    j += ny2 as i32;
                }
                while j >= ny2 as i32 {
                    j -= ny2 as i32;
                }
                if j >= Y as i32 {
                    j = ny2 as i32 - 1 - j;
                }
                val += tmp[x + j as usize * X] * kernel[i];
            }
            out[x + y * X] = val as u8;
        }
    }
}

fn chain(from: usize, to: usize, Ex: &Vec<f64>, Ey: &Vec<f64>, Gx: &Vec<f64>, Gy: &Vec<f64>, X: usize, Y: usize) -> f64 {
    if Ex.is_empty() || Ey.is_empty() || Gx.is_empty() || Gy.is_empty() {
        panic!("chain: invalid input");
    }
    if from >= X * Y || to >= X * Y {
        panic!("chain: one of the points is out of the image");
    }
    // Check that the points are different and valid edge points, otherwise return invalid chaining
    if from == to {
        return 0.0; // same pixel, not a valid chaining
    }
    if Ex[from] < 0.0 || Ey[from] < 0.0 || Ex[to] < 0.0 || Ey[to] < 0.0 {
        return 0.0; // one of them is not an edge point, not a valid chaining
    }
    let dx = Ex[to] - Ex[from];
    let dy = Ey[to] - Ey[from];
    if (Gy[from] * dx - Gx[from] * dy) * (Gy[to] * dx - Gx[to] * dy) <= 0.0 {
        return 0.0;
    }
    if (Gy[from] * dx - Gx[from] * dy) >= 0.0 {
        return 1.0 / dist(Ex[from], Ey[from], Ex[to], Ey[to]);
    } else {
        return -1.0 / dist(Ex[from], Ey[from], Ex[to], Ey[to]);
    }
}

fn compute_gradient(Gx: &mut Vec<f64>, Gy: &mut Vec<f64>, modG: &mut Vec<f64>, image: &Vec<u8>, X: usize, Y: usize) {
    if Gx.is_empty() || Gy.is_empty() || modG.is_empty() || image.is_empty() {
        panic!("compute_gradient: invalid input");
    }
    // Approximate image gradient using centered differences
    for x in 1..(X - 1) {
        for y in 1..(Y - 1) {
            let index = x + y * X;
            Gx[index] = image[(x + 1) + y * X] as f64 - image[(x - 1) + y * X] as f64;
            Gy[index] = image[x + (y + 1) * X] as f64 - image[x + (y - 1) * X] as f64;
            modG[index] = (Gx[index].powi(2) + Gy[index].powi(2)).sqrt();
        }
    }
}

fn compute_edge_points(Ex: &mut Vec<f64>, Ey: &mut Vec<f64>, modG: &Vec<f64>, Gx: &Vec<f64>, Gy: &Vec<f64>, X: usize, Y: usize) {
    if Ex.is_empty() || Ey.is_empty() || modG.is_empty() || Gx.is_empty() || Gy.is_empty() {
        panic!("compute_edge_points: invalid input");
    }
    for i in 0..X * Y {
        Ex[i] = -1.0;
        Ey[i] = -1.0;
    }
    for x in 2..(X - 2) {
        for y in 2..(Y - 2) {
            let mut Dx = 0;
            let mut Dy = 0;
            let mod_v = modG[x + y * X];
            let L = modG[x - 1 + y * X];
            let R = modG[x + 1 + y * X];
            let U = modG[x + (y + 1) * X];
            let D = modG[x + (y - 1) * X];
            let gx = Gx[x + y * X].abs();
            let gy = Gy[x + y * X].abs();
            if greater_round(mod_v, L) && !greater_round(R, mod_v) && gx >= gy {
                Dx = 1;
            } else if greater_round(mod_v, D) && !greater_round(U, mod_v) && gx <= gy {
                Dy = 1;
            }
            if Dx > 0 || Dy > 0 {
                let a = modG[x - Dx + (y - Dy) * X];
                let b = modG[x + y * X];
                let c = modG[x + Dx + (y + Dy) * X];
                let offset = 0.5 * (a - c) / (a - b - b + c);
                Ex[x + y * X] = x as f64 + offset * Dx as f64;
                Ey[x + y * X] = y as f64 + offset * Dy as f64;
            }
        }
    }
}

fn chain_edge_points(next: &mut Vec<i32>, prev: &mut Vec<i32>, Ex: &Vec<f64>, Ey: &Vec<f64>, Gx: &Vec<f64>, Gy: &Vec<f64>, X: usize, Y: usize) {
    if next.is_empty() || prev.is_empty() || Ex.is_empty() || Ey.is_empty() || Gx.is_empty() || Gy.is_empty() {
        panic!("chain_edge_points: invalid input");
    }
    for i in 0..X * Y {
        next[i] = -1;
        prev[i] = -1;
    }
    for x in 2..(X - 2) {
        for y in 2..(Y - 2) {
            if Ex[x + y * X] >= 0.0 && Ey[x + y * X] >= 0.0 {
                let from = x + y * X;
                let mut fwd_s = 0.0;
                let mut bck_s = 0.0;
                let mut fwd = -1i32;
                let mut bck = -1i32;
                for i in 0..5 {
                    for j in 0..5 {
                        let to = x + i - 2 + (y + j - 2) * X;
                        let s = chain(from, to, &Ex, &Ey, &Gx, &Gy, X, Y);
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

                if fwd >= 0 && next[from] != fwd && ((prev[fwd as usize] < 0) || chain(prev[fwd as usize] as usize, fwd as usize, &Ex, &Ey, &Gx, &Gy, X, Y) < fwd_s) {
                    if next[from] >= 0 {
                        prev[next[from] as usize] = -1;
                    }
                    next[from] = fwd;
                    if prev[fwd as usize] >= 0 {
                        next[prev[fwd as usize] as usize] = -1;
                    }
                    prev[fwd as usize] = from as i32;
                }
                if bck >= 0 && prev[from] != bck && ((next[bck as usize] < 0) || chain(next[bck as usize] as usize, bck as usize, &Ex, &Ey, &Gx, &Gy, X, Y) > bck_s) {
                    if next[bck as usize] >= 0 {
                        prev[next[bck as usize] as usize] = -1;
                    }
                    next[bck as usize] = from as i32;
                    if prev[from] >= 0 {
                        next[prev[from] as usize] = -1;
                    }
                    prev[from] = bck;
                }
            }
        }
    }
}

fn thresholds_with_hysteresis(next: &mut Vec<i32>, prev: &mut Vec<i32>, modG: &Vec<f64>, X: usize, Y: usize, th_h: f64, th_l: f64) {
    if next.is_empty() || prev.is_empty() || modG.is_empty() {
        panic!("thresholds_with_hysteresis: invalid input");
    }
    let mut valid: Vec<bool> = vec![false; X * Y];
    for i in 0..X * Y {
        if (prev[i] >= 0 || next[i] >= 0) && !valid[i] && modG[i] >= th_h {
            valid[i] = true;
            let mut j = i;
            while j >= 0 && (next[j] >= 0) && !valid[next[j] as usize] {
                let k = next[j] as usize;
                if modG[k] < th_l {
                    next[j] = -1;
                    prev[k] = -1;
                } else {
                    valid[k] = true;
                }
                j = k as usize;
            }
            j = i;
            while j >= 0 && (prev[j] >= 0) && !valid[prev[j] as usize] {
                let k = prev[j]  as usize;
                if modG[k] < th_l {
                    prev[j] = -1;
                    next[k] = -1;
                } else {
                    valid[k] = true;
                }
                j = k as usize;
            }
        }
    }
    for i in 0..X * Y {
        if (prev[i] >= 0 || next[i] >= 0) && !valid[i] {
            prev[i] = -1;
            next[i] = -1;
        }
    }
}

fn list_chained_edge_points(x: &mut Vec<f64>, y: &mut Vec<f64>, N: &mut i32, curve_limits: &mut Vec<i32>, M: &mut i32, next: &mut Vec<i32>, prev: &mut Vec<i32>, Ex: &Vec<f64>, Ey: &Vec<f64>, X: usize, Y: usize) {
    *x = vec![0.0; X * Y];
    *y = vec![0.0; X * Y];
    *curve_limits = vec![0; X * Y];
    *N = 0;
    *M = 0;
    for i in 0..X * Y {
        if prev[i] >= 0 || next[i] >= 0 {
            curve_limits[*M as usize] = *N;
            *M += 1;
            let mut k = i as i32;
            let mut n;
            'kn: loop {
                n = prev[k as usize];
                if !(n >= 0 && n != i as i32) {
                    break 'kn;
                }
                k = n ;
            }
            while k >= 0 {
                x[*N as usize] = Ex[k as usize];
                y[*N as usize] = Ey[k as usize];
                *N += 1;
                n = next[k as usize];
                next[k as usize] = -1;
                prev[k as usize] = -1;
                k = n ;
            } 
        }
    }
    curve_limits[*M as usize] = *N;
}

pub fn devernay(image: &Vec<u8>, width: usize, height: usize, sigma: f64, th_h: f64, th_l: f64) {
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut N: i32 = 0;
    let mut curve_limits: Vec<i32> = Vec::new();
    let mut M: i32 = 0;
    // let image: Vec<u8> = vec![0; width * height];  // Assuming X and Y are defined
    let mut gauss: Vec<u8> = vec![0; width * height];  // Assuming X and Y are defined
    // let sigma: f64 = 1.0;  // Example value for sigma
    // let th_h: f64 = 0.5;  // Example value for th_h
    // let th_l: f64 = 0.1;  // Example value for th_l
    let mut Gx: Vec<f64> = vec![0.0; width * height];
    let mut Gy: Vec<f64> = vec![0.0; width * height];
    let mut modG: Vec<f64> = vec![0.0; width * height];
    let mut Ex: Vec<f64> = vec![0.0; width * height];
    let mut Ey: Vec<f64> = vec![0.0; width * height];
    let mut next: Vec<i32> = vec![-1; width * height];
    let mut prev: Vec<i32> = vec![-1; width * height];
    if sigma == 0.0 {
        compute_gradient(&mut Gx, &mut Gy, &mut modG, &image, width, height);
    } else {
        gaussian_filter(&image, &mut gauss, width, height, sigma);
        compute_gradient(&mut Gx, &mut Gy, &mut modG, &gauss, width, height);
    }
    // for i in 0..width * height {
    //     if gauss[i] < 255 {
    //         println!("{}, {}", i, gauss[i]);
    //     }
    // }
    for i in 0..width * height {
        if Gx[i] == 0.0 || Gy[i] == 0.0 || modG[i] == 0.0 {
            continue;
        }
    }
    compute_edge_points(&mut Ex, &mut Ey, &modG, &Gx, &Gy, width, height);
    chain_edge_points(&mut next, &mut prev, &Ex, &Ey, &Gx, &Gy, width, height);
    thresholds_with_hysteresis(&mut next, &mut prev, &modG, width, height, th_h, th_l);
    list_chained_edge_points(&mut x, &mut y, &mut N, &mut curve_limits, &mut M, &mut next, &mut prev, &Ex, &Ey, width, height);
    let len = x.len();
    for i in 0..len {
        if x[i]==0f64&& y[i]==0f64 { 
            continue;
        }
        println!("x: {:.4}, y: {:.4}", x[i], y[i]);
    }

}

