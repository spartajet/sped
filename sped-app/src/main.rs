use image::io::Reader as ImageReader;
use log::{info, LevelFilter};
use log4rs::append::console::ConsoleAppender;
use log4rs::config::{Appender, Root};
use log4rs::encode::pattern::PatternEncoder;
use log4rs::{Config, Handle};

use sped::{devernay};

fn main() {
    let _ = config_log4r();
    let origin_mat = ImageReader::open("../resources/demo.Bmp")
        .unwrap()
        .decode()
        .unwrap();
    let gray_mat = origin_mat.grayscale().to_luma8();
    let (width, height) = &gray_mat.dimensions();
    let data: Vec<u8> = gray_mat.into_vec();

    let result = devernay(
        &data,
        width.clone() as usize,
        height.clone() as usize,
        1.5f64,
        4.2f64,
        0.81f64,
    )
    .unwrap();
    info!("Result: {:?}", result.len())
}

fn config_log4r() -> Handle {
    let pattern = "{d(%Y-%m-%d %H:%M:%S %3f)} [{f}:{L}] {h({l})} {m}{n}";
    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(pattern)))
        .build();
    let config = Config::builder()
        .appender(Appender::builder().build("stdout", Box::new(stdout)))
        .build(Root::builder().appender("stdout").build(LevelFilter::Info))
        .unwrap();
    let handle = log4rs::init_config(config).unwrap();
    handle
}
