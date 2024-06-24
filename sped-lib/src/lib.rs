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

fn chain(
    from: usize,
    to: usize,
    ex: &Vec<f64>,
    ey: &Vec<f64>,
    gx: &Vec<f64>,
    gy: &Vec<f64>,
    width: usize,
    height: usize,
) -> f64 {
    if ex.is_empty() || ey.is_empty() || gx.is_empty() || gy.is_empty() {
        panic!("chain: invalid input");
    }
    if from >= width * height || to >= width * height {
        panic!("chain: one of the points is out of the image");
    }
    // Check that the points are different and valid edge points, otherwise return invalid chaining
    if from == to {
        return 0.0; // same pixel, not a valid chaining
    }
    if ex[from] < 0.0 || ey[from] < 0.0 || ex[to] < 0.0 || ey[to] < 0.0 {
        return 0.0; // one of them is not an edge point, not a valid chaining
    }
    let dx = ex[to] - ex[from];
    let dy = ey[to] - ey[from];
    if (gy[from] * dx - gx[from] * dy) * (gy[to] * dx - gx[to] * dy) <= 0.0 {
        return 0.0;
    }
    return if (gy[from] * dx - gx[from] * dy) >= 0.0 {
        1.0 / dist(ex[from], ey[from], ex[to], ey[to])
    } else {
        -1.0 / dist(ex[from], ey[from], ex[to], ey[to])
    };
}

fn compute_gradient(
    image: &Vec<u8>,
    width: usize,
    height: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

fn compute_edge_points(
    mod_g: &Vec<f64>,
    gx: &Vec<f64>,
    gy: &Vec<f64>,
    width: usize,
    height: usize,
) -> (Vec<f64>, Vec<f64>) {
    if mod_g.is_empty() || gx.is_empty() || gy.is_empty() {
        panic!("compute_edge_points: invalid input");
    }
    let len = width * height;
    let mut ex = vec![-1.0; len];
    let mut ey = vec![-1.0; len];
    for x in 2..(width - 2) {
        for y in 2..(height - 2) {
            let mut dx = 0;
            let mut dy = 0;
            let mod_v = mod_g[x + y * width];
            let l = mod_g[x - 1 + y * width];
            let r = mod_g[x + 1 + y * width];
            let u = mod_g[x + (y + 1) * width];
            let d = mod_g[x + (y - 1) * width];
            let gx = gx[x + y * width].abs();
            let gy = gy[x + y * width].abs();
            if greater_round(mod_v, l) && !greater_round(r, mod_v) && gx >= gy {
                dx = 1;
            } else if greater_round(mod_v, d) && !greater_round(u, mod_v) && gx <= gy {
                dy = 1;
            }
            if dx > 0 || dy > 0 {
                let a = mod_g[x - dx + (y - dy) * width];
                let b = mod_g[x + y * width];
                let c = mod_g[x + dx + (y + dy) * width];
                let offset = 0.5 * (a - c) / (a - b - b + c);
                ex[x + y * width] = x as f64 + offset * dx as f64;
                ey[x + y * width] = y as f64 + offset * dy as f64;
            }
        }
    }
    (ex, ey)
}

fn chain_edge_points(
    ex: &Vec<f64>,
    ey: &Vec<f64>,
    gx: &Vec<f64>,
    gy: &Vec<f64>,
    width: usize,
    height: usize,
) -> (Vec<i32>, Vec<i32>) {
    if ex.is_empty() || ey.is_empty() || gx.is_empty() || gy.is_empty() {
        panic!("chain_edge_points: invalid input");
    }
    let len = width * height;
    let mut next = vec![-1i32; len];
    let mut prev = vec![-1i32; len];
    for x in 2..(width - 2) {
        for y in 2..(height - 2) {
            if ex[x + y * width] >= 0.0 && ey[x + y * width] >= 0.0 {
                let from = x + y * width;
                let mut fwd_s = 0.0;
                let mut bck_s = 0.0;
                let mut fwd = -1i32;
                let mut bck = -1i32;
                for i in 0..5 {
                    for j in 0..5 {
                        let to = x + i - 2 + (y + j - 2) * width;
                        let s = chain(from, to, &ex, &ey, &gx, &gy, width, height);
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

                if fwd >= 0
                    && next[from] != fwd
                    && ((prev[fwd as usize] < 0)
                        || chain(
                            prev[fwd as usize] as usize,
                            fwd as usize,
                            &ex,
                            &ey,
                            &gx,
                            &gy,
                            width,
                            height,
                        ) < fwd_s)
                {
                    if next[from] >= 0 {
                        prev[next[from] as usize] = -1;
                    }
                    next[from] = fwd;
                    if prev[fwd as usize] >= 0 {
                        next[prev[fwd as usize] as usize] = -1;
                    }
                    prev[fwd as usize] = from as i32;
                }
                if bck >= 0
                    && prev[from] != bck
                    && ((next[bck as usize] < 0)
                        || chain(
                            next[bck as usize] as usize,
                            bck as usize,
                            &ex,
                            &ey,
                            &gx,
                            &gy,
                            width,
                            height,
                        ) > bck_s)
                {
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
    (prev, next)
}

fn thresholds_with_hysteresis(
    next: &mut Vec<i32>,
    prev: &mut Vec<i32>,
    mod_g: &Vec<f64>,
    width: usize,
    height: usize,
    th_h: f64,
    th_l: f64,
) {
    if next.is_empty() || prev.is_empty() || mod_g.is_empty() {
        panic!("thresholds_with_hysteresis: invalid input");
    }
    let mut valid: Vec<bool> = vec![false; width * height];
    for i in 0..width * height {
        if (prev[i] >= 0 || next[i] >= 0) && !valid[i] && mod_g[i] >= th_h {
            valid[i] = true;
            let mut j = i;
            while j >= 0 && (next[j] >= 0) && !valid[next[j] as usize] {
                let k = next[j] as usize;
                if mod_g[k] < th_l {
                    next[j] = -1;
                    prev[k] = -1;
                } else {
                    valid[k] = true;
                }
                j = k;
            }
            j = i;
            while j >= 0 && (prev[j] >= 0) && !valid[prev[j] as usize] {
                let k = prev[j] as usize;
                if mod_g[k] < th_l {
                    prev[j] = -1;
                    next[k] = -1;
                } else {
                    valid[k] = true;
                }
                j = k;
            }
        }
    }
    for i in 0..width * height {
        if (prev[i] >= 0 || next[i] >= 0) && !valid[i] {
            prev[i] = -1;
            next[i] = -1;
        }
    }
}

fn list_chained_edge_points(
    x: &mut Vec<f64>,
    y: &mut Vec<f64>,
    next: &mut Vec<i32>,
    prev: &mut Vec<i32>,
    ex: &Vec<f64>,
    ey: &Vec<f64>,
    width: usize,
    height: usize,
) -> (i32, i32, Vec<i32>) {
    let mut n = 0;
    let mut m = 0;
    let mut t = 0;
    let mut curve_limits = vec![0; width * height];
    *x = vec![0.0; width * height];
    *y = vec![0.0; width * height];
    for i in 0..width * height {
        if prev[i] >= 0 || next[i] >= 0 {
            curve_limits[m as usize] = n;
            m += 1;
            let mut k = i as i32;
            // let mut n;
            'kn: loop {
                t = prev[k as usize];
                if !(t >= 0 && t != i as i32) {
                    break 'kn;
                }
                k = t;
            }
            while k >= 0 {
                x[n as usize] = ex[k as usize];
                y[n as usize] = ey[k as usize];
                n += 1;
                t = next[k as usize];
                next[k as usize] = -1;
                prev[k as usize] = -1;
                k = t;
            }
        }
    }
    curve_limits[m as usize] = n;
    (m, n, curve_limits)
}

pub fn devernay(
    image: &Vec<u8>,
    width: usize,
    height: usize,
    sigma: f64,
    th_h: f64,
    th_l: f64,
) -> (i32, i32, Vec<i32>, Vec<f64>, Vec<f64>) {
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();

    let cal_image = match sigma == 0f64 {
        true => &image,
        false => &gaussian_filter(&image, width, height, sigma),
    };
    let (gx, gy, mod_g) = compute_gradient(cal_image, width, height);
    for i in 0..width * height {
        if gx[i] == 0.0 || gy[i] == 0.0 || mod_g[i] == 0.0 {
            continue;
        }
    }
    let (ex, ey) = compute_edge_points(&mod_g, &gx, &gy, width, height);
    let (mut prev, mut next) = chain_edge_points(&ex, &ey, &gx, &gy, width, height);
    thresholds_with_hysteresis(&mut next, &mut prev, &mod_g, width, height, th_h, th_l);
    let (m, n, curve_limits) = list_chained_edge_points(
        &mut x, &mut y, &mut next, &mut prev, &ex, &ey, width, height,
    );
    let len = x.len();
    for i in 0..len {
        if x[i] == 0f64 && y[i] == 0f64 {
            continue;
        }
    }
    (m, n, curve_limits, x, y)
}
