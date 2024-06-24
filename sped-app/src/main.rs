use sped_lib::devernay;
use std::io::Cursor;
use image::io::Reader as ImageReader;

fn main() {
    let origin_mat = ImageReader::open("../resources/demo.Bmp").unwrap().decode().unwrap();
    let gray_mat = origin_mat.grayscale().to_luma8();
    let (width,height)=&gray_mat.dimensions();
    let data: Vec<u8> = gray_mat.into_vec();
    
    
    let result = devernay(&data, width.clone() as usize, height.clone() as usize, 1.5f64, 4.2f64, 0.81f64);
}