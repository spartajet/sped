use image::buffer::ConvertBuffer;
use image::{ImageReader, RgbImage};
use log::{info, LevelFilter};
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use log4rs::Config;

use sped::devernay;

fn main() {
    config_log4r();
    let origin_mat = ImageReader::open("../resources/demo1.Bmp")
        .unwrap()
        .decode()
        .unwrap();
    let gray_mat = origin_mat.grayscale().to_luma8();
    let (width, height) = &gray_mat.dimensions();
    let data: Vec<u8> = gray_mat.clone().into_vec();
    let mut result_image: RgbImage = gray_mat.convert();

    let result = devernay(&data, *width as usize, *height as usize, 1.0, 40.0, 20.0).unwrap();
    info!("Result: {:?}", result.len());
    for points in result {
        info!("Points: {:?}", points.len());
        for point in points {
            result_image.put_pixel(point.x as u32, point.y as u32, image::Rgb([255, 0, 0]));
        }
    }
    result_image.save("../resources/result.jpg").unwrap();
}

fn config_log4r() {
    let pattern = "{d(%Y-%m-%d %H:%M:%S %3f)} [{f}:{L}] {h({l})} {m}{n}";
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();
    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(LevelFilter::Info))
        .unwrap();
    log4rs::init_config(config);
}
