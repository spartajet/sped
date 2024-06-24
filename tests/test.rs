#[cfg(test)]
mod tests {
    use opencv::imgcodecs::imread;
    use opencv::prelude::{MatTraitConst, MatTraitConstManual};
    use sped::{devernay, DevernayResult};

    #[test]
    fn edge_detect_test() {
        let origin_mat = imread("resources/demo.Bmp", opencv::imgcodecs::IMREAD_GRAYSCALE).unwrap();
        if origin_mat.is_continuous() { 
            println!("is continuous");
        }else { 
            println!("not continuous");
        }
        let data = origin_mat.data_bytes().unwrap().to_vec();
        let cols = origin_mat.cols() as usize;
        let rows = origin_mat.rows() as usize;
        let result: DevernayResult = devernay(&data, rows, cols, 1.5f64, 4.2f64, 0.81f64).unwrap();
        let len = result.xs.len();
        for i in 0..len {
            let x = result.xs[i];
            let y = result.ys[i];
            println!("x: {:.4}, y: {:.4}", x, y);
        }
    }

    #[test]
    fn usize_test() {
        let a = -1;
        println!("{}", a as usize);
    }
}
