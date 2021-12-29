use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use lenna_core::plugins::PluginRegistrar;
use lenna_core::ProcessorConfig;
use lenna_core::{core::processor::ExifProcessor, core::processor::ImageProcessor, Processor};
use std::io::Cursor;
use tract_ndarray::{ArrayBase, Axis, Dim, ViewRepr};
use tract_onnx::prelude::*;

extern "C" fn register(registrar: &mut dyn PluginRegistrar) {
    registrar.add_plugin(Box::new(UltraFace::default()));
}

lenna_core::export_plugin!(register);

type ModelType = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Clone)]
pub struct UltraFace {
    model: ModelType,
}

impl UltraFace {
    pub fn model() -> ModelType {
        let data = include_bytes!("../assets/version-RFB-640.onnx");
        let mut cursor = Cursor::new(data);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .unwrap()
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 480, 640)),
            )
            .unwrap()
            .into_optimized()
            .unwrap()
            .into_runnable()
            .unwrap();
        model
    }

    pub fn scale(
        width: u32,
        height: u32,
        abox: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    ) -> Rect {
        let width = width as f32;
        let height = height as f32;
        let x = abox[0];
        let y = abox[1];
        let w = abox[2] - abox[0];
        let h = abox[3] - abox[1];

        Rect::at((width * x) as i32, (height * y) as i32)
            .of_size(((width * w).abs()) as u32, ((height * h).abs()) as u32)
    }
}

impl Default for UltraFace {
    fn default() -> Self {
        UltraFace {
            model: Self::model(),
        }
    }
}

impl ImageProcessor for UltraFace {
    fn process_image(
        &self,
        image: &mut Box<DynamicImage>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let image_rgb = image.to_rgb8();
        let resized = image::imageops::resize(
            &image_rgb,
            640,
            480,
            ::image::imageops::FilterType::Triangle,
        );
        let tensor: Tensor =
            tract_ndarray::Array4::from_shape_fn((1, 3, 480, 640), |(_, c, y, x)| {
                let mean = [0.485, 0.456, 0.406][c];
                let std = [0.229, 0.224, 0.225][c];
                (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
            })
            .into();

        let result = self.model.run(tvec!(tensor)).unwrap();
        let scores: tract_ndarray::ArrayView3<f32> =
            result[0].to_array_view::<f32>()?.into_dimensionality()?;
        let scores = scores.index_axis(Axis(0), 0);

        let boxes: tract_ndarray::ArrayView3<f32> =
            result[1].to_array_view::<f32>()?.into_dimensionality()?;
        let boxes = boxes.index_axis(Axis(0), 0);

        let threshold = 0.9;
        let white = Rgba([255u8, 255u8, 255u8, 255u8]);

        let mut img = DynamicImage::ImageRgba8(image.to_rgba8());
        let (width, height) = img.dimensions();

        for (score, abox) in scores.rows().into_iter().zip(boxes.rows()) {
            if score[1] > threshold {
                let rect = Self::scale(width, height, abox);
                draw_hollow_rect_mut(&mut img, rect, white);
            }
        }
        *image = Box::new(img);
        Ok(())
    }
}

impl ExifProcessor for UltraFace {}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Config {}

impl Default for Config {
    fn default() -> Self {
        Config {}
    }
}

impl Processor for UltraFace {
    fn name(&self) -> String {
        "ultraface".into()
    }

    fn title(&self) -> String {
        "UltraFace".into()
    }

    fn author(&self) -> String {
        "chriamue".into()
    }

    fn description(&self) -> String {
        "Plugin to detect faces in images.".into()
    }

    fn process(
        &mut self,
        _config: ProcessorConfig,
        image: &mut Box<lenna_core::LennaImage>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.process_exif(&mut image.exif).unwrap();
        self.process_image(&mut image.image).unwrap();
        Ok(())
    }

    fn default_config(&self) -> serde_json::Value {
        serde_json::to_value(Config::default()).unwrap()
    }
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
lenna_core::export_wasm_plugin!(UltraFace);

#[cfg(test)]
mod tests {
    use super::*;
    use lenna_core::ProcessorConfig;

    #[test]
    fn default() {
        let mut ultraface = UltraFace::default();
        let config = ProcessorConfig {
            id: "mobilenet".into(),
            config: ultraface.default_config(),
        };
        assert_eq!(ultraface.name(), "ultraface");
        let mut img =
            Box::new(lenna_core::io::read::read_from_file("assets/lenna.png".into()).unwrap());
        ultraface.process(config, &mut img).unwrap();
        img.name = "test".to_string();
        lenna_core::io::write::write_to_file(&img, image::ImageOutputFormat::Jpeg(80)).unwrap();
    }

    #[cfg(target_arch = "wasm32")]
    mod wasm {
        use super::*;
        use wasm_bindgen_test::*;

        #[wasm_bindgen_test]
        fn default() {
            let ultraface = UltraFace::default();

            assert_eq!(ultraface.name(), "ultraface");
        }
    }
}
