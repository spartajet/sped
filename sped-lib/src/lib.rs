use std::f64::consts::E;

const EPSILON: f64 = f64::EPSILON;

fn greater_round(a: f64, b: f64) -> bool {
    if a <= b {
        return false;
    }
    if (a - b) < EPSILON {
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
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

fn gaussian_filter(image: &Vec<u8>, width: usize, height: usize, sigma: f64) -> Vec<u8> {
    let mut tmp = vec![0.0; width * height];
    let mut out = vec![0; width * height];
    let prec = 3.0;
    let offset = (sigma * (2.0 * prec * 10f64.ln()).sqrt()).ceil() as usize;
    let n = 1 + 2 * offset;
    let kernel = gaussian_kernel(n, sigma, offset as f64);
    let nx2 = 2 * width;
    let ny2 = 2 * height;
    for x in 0..width {
        for y in 0..height {
            let mut val = 0.0;
            for i in 0..n {
                let mut j = x as i32 - offset as i32 + i as i32;
                while j < 0 {
                    j += nx2 as i32;
                }
                while j >= nx2 as i32 {
                    j -= nx2 as i32;
                }
                if j >= width as i32 {
                    j = nx2 as i32 - 1 - j;
                }
                val += (image[j as usize + y * width] as f64) * kernel[i];
            }
            tmp[x + y * width] = val;
        }
    }
    for x in 0..width {
        for y in 0..height {
            let mut val = 0.0;
            for i in 0..n {
                let mut j = y as i32 - offset as i32 + i as i32;
                while j < 0 {
                    j += ny2 as i32;
                }
                while j >= ny2 as i32 {
                    j -= ny2 as i32;
                }
                if j >= height as i32 {
                    j = ny2 as i32 - 1 - j;
                }
                val += tmp[x + j as usize * width] * kernel[i];
            }
            out[x + y * width] = val as u8;
        }
    }
    out
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

fn compute_gradient(image: &Vec<u8>, width: usize, height: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Approximate image gradient using centered differences
    let len = width * height;
    let mut gx = vec![0f64; len];
    let mut gy = vec![0f64; len];
    let mut mod_g = vec![0f64; len];
    for x in 1..(width - 1) {
        for y in 1..(height - 1) {
            let index = x + y * width;
            gx[index] = image[(x + 1) + y * width] as f64 - image[(x - 1) + y * width] as f64;
            gy[index] = image[x + (y + 1) * width] as f64 - image[x + (y - 1) * width] as f64;
            mod_g[index] = (gx[index].powi(2) + gy[index].powi(2)).sqrt();
        }
    }
    (gx, gy, mod_g)
}

fn compute_edge_points(mod_g: &Vec<f64>, gx: &Vec<f64>, gy: &Vec<f64>, width: usize, height: usize) -> (Vec<f64>, Vec<f64>) {
    if mod_g.is_empty() || gx.is_empty() || gy.is_empty() {
        panic!("compute_edge_points: invalid input");
    }
    let len = width * height;
    let mut ex = vec![-1.0; len];
    let mut ey = vec![-1.0; len];
    for x in 2..(width - 2) {
        for y in 2..(height - 2) {
            let mut Dx = 0;
            let mut Dy = 0;
            let mod_v = mod_g[x + y * width];
            let L = mod_g[x - 1 + y * width];
            let R = mod_g[x + 1 + y * width];
            let U = mod_g[x + (y + 1) * width];
            let D = mod_g[x + (y - 1) * width];
            let gx = gx[x + y * width].abs();
            let gy = gy[x + y * width].abs();
            if greater_round(mod_v, L) && !greater_round(R, mod_v) && gx >= gy {
                Dx = 1;
            } else if greater_round(mod_v, D) && !greater_round(U, mod_v) && gx <= gy {
                Dy = 1;
            }
            if Dx > 0 || Dy > 0 {
                let a = mod_g[x - Dx + (y - Dy) * width];
                let b = mod_g[x + y * width];
                let c = mod_g[x + Dx + (y + Dy) * width];
                let offset = 0.5 * (a - c) / (a - b - b + c);
                ex[x + y * width] = x as f64 + offset * Dx as f64;
                ey[x + y * width] = y as f64 + offset * Dy as f64;
            }
        }
    }
    (ex, ey)
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
                let k = prev[j] as usize;
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
                k = n;
            }
            while k >= 0 {
                x[*N as usize] = Ex[k as usize];
                y[*N as usize] = Ey[k as usize];
                *N += 1;
                n = next[k as usize];
                next[k as usize] = -1;
                prev[k as usize] = -1;
                k = n;
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
    // let mut Ex: Vec<f64> = vec![0.0; width * height];
    // let mut Ey: Vec<f64> = vec![0.0; width * height];
    let mut next: Vec<i32> = vec![-1; width * height];
    let mut prev: Vec<i32> = vec![-1; width * height];

    let cal_image = match sigma == 0f64 {
        true => &image,
        false => &gaussian_filter(&image, width, height, sigma)
    };
    let (gx, gy, mod_g) = compute_gradient(cal_image, width, height);
    for i in 0..width * height {
        if gx[i] == 0.0 || gy[i] == 0.0 || mod_g[i] == 0.0 {
            continue;
        }
    }
    let (ex, ey) = compute_edge_points(&mod_g, &gx, &gy, width, height);
    chain_edge_points(&mut next, &mut prev, &ex, &ey, &gx, &gy, width, height);
    thresholds_with_hysteresis(&mut next, &mut prev, &mod_g, width, height, th_h, th_l);
    list_chained_edge_points(&mut x, &mut y, &mut N, &mut curve_limits, &mut M, &mut next, &mut prev, &ex, &ey, width, height);
    println!("N: {}, M: {}", N, M);
    let len = x.len();
    for i in 0..len {
        if x[i] == 0f64 && y[i] == 0f64 {
            continue;
        }
        // println!("x: {:.4}, y: {:.4}", x[i], y[i]);
    }
}

