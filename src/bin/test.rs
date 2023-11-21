use std::io::{Cursor, Read, Seek, SeekFrom};

use image::{EncodableLayout, Rgb, RgbImage};
use imageproc::drawing::draw_filled_circle_mut;
use minimp4::Mp4Muxer;

use openh264::encoder::{Encoder, EncoderConfig};

fn main() {
    let config = EncoderConfig::new(512, 512);
    let mut encoder = Encoder::with_config(config).unwrap();

    let mut buf = Vec::new();

    for i in 0..512 {
        let frame = get_next_frame(i);
        // Convert RGB into YUV.
        let yuv = openh264::formats::YUVBuffer::with_rgb(512, 512, &frame[..]);

        // Encode YUV into H.264.
        let bitstream = encoder.encode(&yuv).unwrap();
        bitstream.write_vec(&mut buf);
    }

    let mut video_buffer = Cursor::new(Vec::new());
    let mut mp4muxer = Mp4Muxer::new(&mut video_buffer);
    mp4muxer.init_video(512, 512, false, "Moving circle.");
    mp4muxer.write_video(&buf);
    mp4muxer.close();

    // Some shenanigans to get the raw bytes for the video.
    video_buffer.seek(SeekFrom::Start(0)).unwrap();
    let mut video_bytes = Vec::new();
    video_buffer.read_to_end(&mut video_bytes).unwrap();

    std::fs::write("circle.mp4", &video_bytes).unwrap();
}

fn get_next_frame(index: u32) -> Vec<u8> {
    let red = Rgb([255u8, 0u8, 0u8]);

    let mut image = RgbImage::new(512, 512);
    draw_filled_circle_mut(&mut image, (index as i32, index as i32), 40, red);

    image.as_bytes().to_vec()
}
