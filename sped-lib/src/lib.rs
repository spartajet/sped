#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use anyhow::{anyhow, Result};
use log::trace;
use std::f64::consts::E;

const EPSILON: f64 = f64::EPSILON;

/// compute a > b considering the rounding errors due to the representation of double numbers
///
/// # Arguments
///
/// * `a`:
/// * `b`:
///
/// returns: bool
///
/// # Examples
///
/// ```
///
/// ```
fn greater_round(a: f64, b: f64) -> bool {
    if a <= b {
        return false;
    }
    if (a - b) < EPSILON {
        return false;
    }
    true
}

///
/// compute a Gaussian kernel of length n, standard deviation sigma,and centered at value mean.
/// The size of the kernel is selected to guarantee that the first discarded term is at least 10^prec times
/// smaller than the central value.For that,the half size of the kernel must be larger than x, with：
/// e^(-x^2/2sigma^2) = 1/10^prec
/// Then,x = sigma * sqrt( 2 * prec * ln(10) ).
///
/// for example, if mean=0.5, the Gaussian will be centered in the middle point between values kernel[0] and kernel[1].
/// kernel must be allocated to a size n.
/// # Arguments
///
/// * `n`:
/// * `sigma`:
/// * `mean`:
///
/// returns: Vec<f64, Global>
///
/// # Examples
///
/// ```
///
/// ```
fn gaussian_kernel(n: usize, sigma: f64, mean: f64) -> Vec<f64> {
    let mut kernel = Vec::with_capacity(n);
    for i in 0..n {
        let val = ((i as f64) - mean) / sigma;
        kernel.push(E.powf(-0.5 * val * val));
    }
    let sum: f64 = kernel.iter().sum();
    kernel.iter().map(|x| x / sum).collect()
}

/// Compute the L2 distance between two points.
///
/// # Arguments
///
/// * `x1`: The x-coordinate of the first point.
/// * `y1`: The y-coordinate of the first point.
/// * `x2`: The x-coordinate of the second point.
/// * `y2`: The y-coordinate of the second point.
///
/// returns: f64
///
/// # Examples
///
/// ```
///
/// ```
fn dist_l2(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

///
/// filter an image with a Gaussian kernel of parameter sigma. return a pointer
/// to a newly allocated filtered image, of the same size as the input image.
/// # Arguments
///
/// * `image`: The input image.
/// * `width`: The width of the input image.
/// * `height`: The height of the input image.
/// * `sigma`: The standard deviation of the Gaussian distribution.
///
/// returns: Vec<u8, Global>
///
/// # Examples
///
/// ```
///
/// ```
fn gaussian_filter(image: &[u8], width: usize, height: usize, sigma: f64) -> Vec<u8> {
    let len = width * height;
    let mut tmp = vec![0.0; len];
    let mut out = vec![0; len];
    let precision = 3.0;
    let offset = (sigma * (2.0 * precision * 10f64.ln()).sqrt()).ceil() as usize;
    let n = 1 + 2 * offset;
    let kernel = gaussian_kernel(n, sigma, offset as f64);
    // auxiliary variables for the double of the image size
    let nx2 = 2 * width;
    let ny2 = 2 * height;
    // x axis convolution
    for x in 0..width {
        for y in 0..height {
            let mut val = 0.0;
            for i in 0..n {
                let mut j = x as i32 - offset as i32 + i as i32;
                // symmetry boundary condition
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
    // y axis convolution
    for x in 0..width {
        for y in 0..height {
            let mut val = 0.0;
            for i in 0..n {
                let mut j = y as i32 - offset as i32 + i as i32;
                // symmetry boundary condition
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

///  return a score for chaining pixels 'from' to 'to', favoring closet point:
///  = 0.0 invalid chaining;
///  \> 0.0 valid forward chaining; the larger the value, the better the chaining;
///  \< 0.0 valid backward chaining; the smaller the value, the better the chaining;
///
///
/// # Arguments
///
/// * `from`: the two pixel IDs to evaluate their potential chaining
/// * `to`: the two pixel IDs to evaluate their potential chaining
/// * `ex`: the sub-pixel position of point i, if i is an edge point;
/// * `ey`: the sub-pixel position of point i, if i is an edge point;
/// * `gx`: the image gradient at pixel i;
/// * `gy`: the image gradient at pixel i;
/// * `width`: the size of the image;
/// * `height`: the size of the image;
///
/// returns: f64
///
/// # Examples
///
/// ```
///
/// ```
fn chain(
    from: usize,
    to: usize,
    ex: &[f64],
    ey: &[f64],
    gx: &[f64],
    gy: &[f64],
    width: usize,
    height: usize,
) -> f64 {
    if ex.is_empty() || ey.is_empty() || gx.is_empty() || gy.is_empty() {
        panic!("chain: invalid input");
    }
    let len = width * height;
    if from >= len || to >= len {
        panic!("chain: one of the points is out of the image");
    }
    // Check that the points are different and valid edge points, otherwise return invalid chaining
    if from == to {
        return 0.0; // same pixel, not a valid chaining
    }
    if ex[from] < 0.0 || ey[from] < 0.0 || ex[to] < 0.0 || ey[to] < 0.0 {
        return 0.0; // one of them is not an edge point, not a valid chaining
    }
    /* in a good chaining, the gradient should be roughly orthogonal
    to the line joining the two points to be chained:
    when Gy * dx - Gx * dy > 0, it corresponds to a forward chaining,
    when Gy * dx - Gx * dy < 0, it corresponds to a backward chaining.

    first check that the gradient at both points to be chained agree
    in one direction, otherwise return invalid chaining. */
    let dx = ex[to] - ex[from];
    let dy = ey[to] - ey[from];
    if (gy[from] * dx - gx[from] * dy) * (gy[to] * dx - gx[to] * dy) <= 0.0 {
        return 0.0; /* incompatible gradient angles, not a valid chaining */
    }
    /* return the chaining score: positive for forward chaining,negative for backwards.
    the score is the inverse of the distance to the chaining point, to give preference to closer points */
    if (gy[from] * dx - gx[from] * dy) >= 0.0 {
        1.0 / dist_l2(ex[from], ey[from], ex[to], ey[to]) /* forward chaining  */
    } else {
        -1.0 / dist_l2(ex[from], ey[from], ex[to], ey[to]) /* backward chaining */
    }
}

/// compute the image gradient, giving its x and y components as well
/// as the modulus. Gx, Gy, and modG must be already allocated.
///
/// # Arguments
///
/// * `image`: image data
/// * `width`: image width
/// * `height`: image height
///
/// returns: (Vec<f64, Global>, Vec<f64, Global>, Vec<f64, Global>)
///
/// # Examples
///
/// ```
///
/// ```
fn compute_gradient(image: &[u8], width: usize, height: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = width * height;
    let mut gx = vec![0f64; len];
    let mut gy = vec![0f64; len];
    let mut mod_g = vec![0f64; len];
    // approximate image gradient using centered differences
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

/// compute sub-pixel edge points using adapted Canny and Devernay methods.
/// Devernay correction works very well and is very consistent when used to interpolate only along horizontal
/// or vertical direction. this modified version requires that a pixel, to be an edge point, must be a local maximum
/// horizontally or vertically, depending on the gradient orientation: if the x component of the gradient is larger
/// than the y component, Gx > Gy, this means that the gradient is roughly horizontal and a horizontal maximum is
/// required to be an edge point.
/// # Arguments
///
/// * `mod_g`: modulus of the image gradient
/// * `gx`: x components  of the image gradient.
/// * `gy`:  y components  of the image gradient.
/// * `width`:
/// * `height`:
///
/// returns: (Vec<f64, Global>, Vec<f64, Global>)
///
/// # Examples
///
/// ```
///
/// ```
fn compute_edge_points(
    mod_g: &[f64],
    gx: &[f64],
    gy: &[f64],
    width: usize,
    height: usize,
) -> (Vec<f64>, Vec<f64>) {
    if mod_g.is_empty() || gx.is_empty() || gy.is_empty() {
        panic!("compute_edge_points: invalid input");
    }
    let len = width * height;
    let mut ex = vec![-1.0; len];
    let mut ey = vec![-1.0; len];
    /* explore pixels inside a 2 pixel margin (so modG[x,y +/- 1,1] is defined) */
    for x in 2..(width - 2) {
        for y in 2..(height - 2) {
            let mut dx = 0; /* interpolation is along Dx,Dy		*/
            let mut dy = 0; /* which will be selected below		*/
            let mod_v = mod_g[x + y * width]; /* modG at pixel					*/
            let l = mod_g[x - 1 + y * width]; /* modG at pixel on the left			*/
            let r = mod_g[x + 1 + y * width]; /* modG at pixel on the right		*/
            let u = mod_g[x + (y + 1) * width]; /* modG at pixel up					*/
            let d = mod_g[x + (y - 1) * width]; /* modG at pixel below				*/
            let gx = gx[x + y * width].abs(); /* absolute value of Gx				*/
            let gy = gy[x + y * width].abs(); /* absolute value of Gy				*/
            /* when local horizontal maxima of the gradient modulus and the gradient direction
            is more horizontal (|Gx| >= |Gy|),=> a "horizontal" (H) edge found else,
            if local vertical maxima of the gradient modulus and the gradient direction is more
            vertical (|Gx| <= |Gy|),=> a "vertical" (V) edge found */

            /* it can happen that two neighbor pixels have equal value and are both	maxima, for example
            when the edge is exactly between both pixels. in such cases, as an arbitrary convention,
            the edge is marked on the left one when an horizontal max or below when a vertical max.
            for	this the conditions are L < mod >= R and D < mod >= U,respectively. the comparisons are
            done using the function greater() instead of the operators > or >= so numbers differing only
            due to rounding errors are considered equal */
            if greater_round(mod_v, l) && !greater_round(r, mod_v) && gx >= gy {
                dx = 1; /* H */
            } else if greater_round(mod_v, d) && !greater_round(u, mod_v) && gx <= gy {
                dy = 1;
            } /* V */
            /* Devernay sub-pixel correction

            the edge point position is selected as the one of the maximum of a quadratic interpolation of the magnitude of
            the gradient along a unidimensional direction. the pixel must be a local maximum. so we	have the values:

            the x position of the maximum of the parabola passing through(-1,a), (0,b), and (1,c) is
            offset = (a - c) / 2(a - 2b + c),and because b >= a and b >= c, -0.5 <= offset <= 0.5	*/
            if dx > 0 || dy > 0 {
                /* offset value is in [-0.5, 0.5] */
                let a = mod_g[x - dx + (y - dy) * width];
                let b = mod_g[x + y * width];
                let c = mod_g[x + dx + (y + dy) * width];
                let offset = 0.5 * (a - c) / (a - b - b + c);
                /* store edge point */
                ex[x + y * width] = x as f64 + offset * dx as f64;
                ey[x + y * width] = y as f64 + offset * dy as f64;
            }
        }
    }
    (ex, ey)
}

/// chain edge points
///
/// # Arguments
///
/// * `ex`: the sub-pixel coordinates when an edge point is present or -1,-1 otherwise.
/// * `ey`: the sub-pixel coordinates when an edge point is present or -1,-1 otherwise.
/// * `gx`: x and y components and the modulus of the image gradient
/// * `gy`: x and y components and the modulus of the image gradient
/// * `width`:
/// * `height`:
///
/// returns: (Vec<i32, Global>, Vec<i32, Global>)
/// next and prev:contain the number of next and previous edge points in the chain.
/// when not chained in one of the directions, the corresponding value is set to -1.
/// next and prev must be allocated before calling
/// # Examples
///
/// ```
///
/// ```
fn chain_edge_points(
    ex: &[f64],
    ey: &[f64],
    gx: &[f64],
    gy: &[f64],
    width: usize,
    height: usize,
) -> (Vec<i32>, Vec<i32>) {
    if ex.is_empty() || ey.is_empty() || gx.is_empty() || gy.is_empty() {
        panic!("chain_edge_points: invalid input");
    }
    let len = width * height;
    let mut next = vec![-1i32; len];
    let mut prev = vec![-1i32; len];
    /* try each point to make local chains */
    for x in 2..(width - 2) {
        /* 2 pixel margin to include the tested neighbors */
        for y in 2..(height - 2) {
            if ex[x + y * width] >= 0.0 && ey[x + y * width] >= 0.0 {
                /* must be an edge point */
                let from = x + y * width; /* edge point to be chained			*/
                let mut fwd_s = 0.0; /* score of best forward chaining		*/
                let mut bck_s = 0.0; /* score of best backward chaining		*/
                let mut fwd = -1i32; /* edge point of best forward chaining */
                let mut bck = -1i32; /* edge point of best backward chaining*/
                /* try all neighbors two pixels apart or less.
                looking for candidates for chaining two pixels apart, in most such cases,
                is enough to obtain good chains of edge points that	accurately describes the edge.	*/
                for i in 0..5 {
                    for j in 0..5 {
                        let to = x + i - 2 + (y + j - 2) * width; /* candidate edge point to be chained */
                        let s = chain(from, to, ex, ey, gx, gy, width, height); /* score from-to */
                        if s > fwd_s {
                            /* a better forward chaining found    */
                            fwd_s = s; /* set the new best forward chaining  */
                            fwd = to as i32;
                        }
                        if s < bck_s {
                            /* a better backward chaining found	  */
                            bck_s = s; /* set the new best backward chaining */
                            bck = to as i32;
                        }
                    }
                }
                /* before making the new chain, check whether the target was
                already chained and in that case, whether the alternative
                chaining is better than the proposed one.

                x alt                        x alt
                \                          /
                \                        /
                from x---------x fwd              bck x---------x from

                we know that the best forward chain starting at from is from-fwd.
                but it is possible that there is an alternative chaining arriving
                at fwd that is better, such that alt-fwd is to be preferred to
                from-fwd. an analogous situation is possible in backward chaining,
                where an alternative link bck-alt may be better than bck-from.

                before making the new link, check if fwd/bck are already chained,
                and in such case compare the scores of the proposed chaining to
                the existing one, and keep only the best of the two.

                there is an undesirable aspect of this procedure: the result may
                depend on the order of exploration. consider the following
                configuration:

                a x-------x b
                /
                /
                c x---x d    with score(a-b) < score(c-b) < score(c-d)
                or equivalently ||a-b|| > ||b-c|| > ||c-d||

                let us consider two possible orders of exploration.

                order: a,b,c
                we will first chain a-b when exploring a. when analyzing the
                backward links of b, we will prefer c-b, and a-b will be unlinked.
                finally, when exploring c, c-d will be preferred and c-b will be
                unlinked. the result is just the chaining c-d.

                order: c,b,a
                we will first chain c-d when exploring c. then, when exploring
                the backward connections of b, c-b will be the preferred link;
                but because c-d exists already and has a better score, c-b
                cannot be linked. finally, when exploring a, the link a-b will
                be created because there is no better backward linking of b.
                the result is two chainings: c-d and a-b.

                we did not found yet a simple algorithm to solve this problem. by
                simple, we mean an algorithm without two passes or the need to
                re-evaluate the chaining of points where one link is cut.

                for most edge points, there is only one possible chaining and this
                problem does not arise. but it does happen and a better solution
                is desirable.
                */
                if fwd >= 0
                    && next[from] != fwd
                    && ((prev[fwd as usize] < 0)
                        || chain(
                            prev[fwd as usize] as usize,
                            fwd as usize,
                            ex,
                            ey,
                            gx,
                            gy,
                            width,
                            height,
                        ) < fwd_s)
                {
                    if next[from] >= 0 {
                        /* remove previous from-x link if one */
                        prev[next[from] as usize] = -1; /* only prev requires explicit reset  */
                    }
                    next[from] = fwd; /* set next of from-fwd link          */
                    if prev[fwd as usize] >= 0 {
                        /* remove alt-fwd link if one         */
                        next[prev[fwd as usize] as usize] = -1; /* only next requires explicit reset  */
                    }
                    prev[fwd as usize] = from as i32; /* set prev of from-fwd link          */
                }
                if bck >= 0
                    && prev[from] != bck
                    && ((next[bck as usize] < 0)
                        || chain(
                            next[bck as usize] as usize,
                            bck as usize,
                            ex,
                            ey,
                            gx,
                            gy,
                            width,
                            height,
                        ) > bck_s)
                {
                    if next[bck as usize] >= 0 {
                        /* remove bck-alt link if one         */
                        prev[next[bck as usize] as usize] = -1; /* only prev requires explicit reset  */
                    }
                    next[bck as usize] = from as i32; /* set next of bck-from link          */
                    if prev[from] >= 0 {
                        /* remove previous x-from link if one */
                        next[prev[from] as usize] = -1; /* only next requires explicit reset  */
                    }
                    prev[from] = bck; /* set prev of bck-from link          */
                }
            }
        }
    }
    (prev, next)
}

/// apply Canny thresholding with hysteresis
///
/// next and prev contain the number of next and previous edge points in the
/// chain or -1 when not chained. modG is modulus of the image gradient. X,Y is
/// the image size. th_h and th_l are the high and low thresholds, respectively.
///
/// this function modifies next and prev, removing chains not satisfying the
/// thresholds.
///
/// # Arguments
///
/// * `next`:
/// * `prev`:
/// * `mod_g`:
/// * `width`:
/// * `height`:
/// * `th_h`:
/// * `th_l`:
///
/// returns: ()
///
/// # Examples
///
/// ```
///
/// ```
fn thresholds_with_hysteresis(
    next: &mut [i32],
    prev: &mut [i32],
    mod_g: &[f64],
    width: usize,
    height: usize,
    th_h: f64,
    th_l: f64,
) {
    if next.is_empty() || prev.is_empty() || mod_g.is_empty() {
        panic!("thresholds_with_hysteresis: invalid input");
    }
    let len = width * height;
    let mut valid: Vec<bool> = vec![false; len];
    /* validate all edge points over th_h or connected to them and over th_l */
    for i in 0..len {
        /* prev[i]>=0 or next[i]>=0 implies an edge point */
        if (prev[i] >= 0 || next[i] >= 0) && !valid[i] && mod_g[i] >= th_h {
            valid[i] = true; /* mark as valid the new point */
            let mut j = i;
            /* follow the chain of edge points forwards */
            while (next[j] >= 0) && !valid[next[j] as usize] {
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
            while (prev[j] >= 0) && !valid[prev[j] as usize] {
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
    for i in 0..len {
        if (prev[i] >= 0 || next[i] >= 0) && !valid[i] {
            prev[i] = -1;
            next[i] = -1;
        }
    }
}

///
/// create a list of chained edge points composed of 3 lists
/// x, y and curve_limits; it also computes N (the number of edge points) and
/// M (the number of curves).
///
/// x[i] and y[i] (0<=i<N) store the sub-pixel coordinates of the edge points.
/// curve_limits[j] (0<=j<=M) stores the limits of each chain in lists x and y.
///
/// x, y, and curve_limits will be allocated.
///
/// example:
///
/// curve number k (0<=k<M) consists of the edge points x[i],y[i]
/// for i determined by curve_limits[k] <= i < curve_limits[k+1].
///
/// curve k is closed if x[curve_limits[k]] == x[curve_limits[k+1] - 1] and
/// y[curve_limits[k]] == y[curve_limits[k+1] - 1].
///
/// # Arguments
///
/// * `next`:
/// * `prev`:
/// * `ex`:
/// * `ey`:
/// * `width`:
/// * `height`:
///
/// returns: Vec<Vec<Point2d, Global>, Global>
///
/// # Examples
///
/// ```
///
/// ```
fn list_chained_edge_points(
    next: &mut [i32],
    prev: &mut [i32],
    ex: &[f64],
    ey: &[f64],
    width: usize,
    height: usize,
) -> Vec<Vec<Point2d>> {
    let len = width * height;
    /* initialize output: x, y, curve_limits, N, and M

    there cannot be more than X*Y edge points to be put in the output list:
    edge points must be local maxima of gradient modulus, so at most half of
    the pixels could be so. when a closed curve is found, one edge point will
    be put twice to the output. even if all possible edge points (half of the
    pixels in the image) would form one pixel closed curves (which is not
    possible) that would lead to output X*Y edge points.

    for the same reason, there cannot be more than X*Y curves: the worst case
    is when all possible edge points (half of the pixels in the image) would
    form one pixel chains. in that case (which is not possible) one would need
    a size for curve_limits of X*Y/2+1. so X*Y is enough.

    (curve_limits requires one more item than the number of curves.
    a simplest example is when only one chain of length 3 is present:
    curve_limits[0] = 0, curve_limits[1] = 3.)
    */
    let mut results: Vec<Vec<Point2d>> = Vec::new();

    for i in 0..len {
        if prev[i] >= 0 || next[i] >= 0 {
            let mut result: Vec<Point2d> = Vec::new();
            let mut k = i as i32;
            let mut t: i32;
            'kn: loop {
                t = prev[k as usize];
                if !(t >= 0 && t != i as i32) {
                    break 'kn;
                }
                k = t;
            }
            while k >= 0 {
                result.push(Point2d {
                    x: ex[k as usize],
                    y: ey[k as usize],
                });
                t = next[k as usize];
                next[k as usize] = -1;
                prev[k as usize] = -1;
                k = t;
            }
            results.push(result);
        }
    }

    results
}

///
/// chained, sub-pixel edge detector. based on a modified Canny non-maximal
/// suppression and a modified Devernay sub-pixel correction.
/// # Arguments
///
/// * `image`: the input image
/// * `width`:
/// * `height`:
/// * `sigma`: standard deviation sigma for the Gaussian filtering
/// * `th_h`: high gradient threshold in Canny's hysteresis
/// * `th_l`: low gradient threshold in Canny's hysteresis
///
/// returns: Result<Vec<Vec<Point2d, Global>, Global>, String>
///
/// # Examples
///
/// ```
/// let origin_mat = ImageReader::open("../resources/demo1.Bmp")
///    .unwrap()
///    .decode()
///    .unwrap();
/// let gray_mat = origin_mat.grayscale().to_luma8();
/// let (width, height) = &gray_mat.dimensions();
/// let data: Vec<u8> = gray_mat.clone().into_vec();
/// let mut result_image: RgbImage = gray_mat.convert();
///
/// let result = devernay(&data, *width as usize, *height as usize, 1.0, 40.0, 20.0).unwrap();
/// trace!("Result: {:?}", result.len());
/// for points in result {
///     trace!("Points: {:?}", points.len());
///     for point in points {
///         result_image.put_pixel(point.x as u32, point.y as u32, image::Rgb([255, 0, 0]));
///     }
/// }
/// result_image.save("../resources/result.jpg").unwrap();
///
/// ```
pub fn devernay(
    image: &[u8],
    width: usize,
    height: usize,
    sigma: f64,
    th_h: f64,
    th_l: f64,
) -> Result<Vec<Vec<Point2d>>> {
    if sigma < 0.0 {
        return Err(anyhow!("sigma must be non-negative"));
    }
    if th_h < 0.0 || th_l < 0.0 {
        return Err(anyhow!("thresholds must be non-negative"));
    }
    if th_h < th_l {
        return Err(anyhow!("high threshold must be greater than low threshold"));
    }

    let (gx, gy, mod_g) = if sigma == 0f64 {
        compute_gradient(image, width, height)
    } else {
        let gaussian_mat = gaussian_filter(image, width, height, sigma);
        compute_gradient(&gaussian_mat, width, height)
    };
    let (ex, ey) = compute_edge_points(&mod_g, &gx, &gy, width, height);
    let (mut prev, mut next) = chain_edge_points(&ex, &ey, &gx, &gy, width, height);
    thresholds_with_hysteresis(&mut next, &mut prev, &mod_g, width, height, th_h, th_l);
    let results = list_chained_edge_points(&mut next, &mut prev, &ex, &ey, width, height);
    trace!("Devernay result: {:?}", results.len());
    Ok(results)
}

/// Represents a 2D point with floating-point coordinates.
///
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}
