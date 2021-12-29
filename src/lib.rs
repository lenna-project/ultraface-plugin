use image::DynamicImage;
use lenna_core::plugins::PluginRegistrar;
use lenna_core::ProcessorConfig;
use lenna_core::{core::processor::ExifProcessor, core::processor::ImageProcessor, Processor};

extern "C" fn register(registrar: &mut dyn PluginRegistrar) {
    registrar.add_plugin(Box::new(UltraFace::default()));
}

lenna_core::export_plugin!(register);

#[derive(Clone)]
pub struct UltraFace {}

impl Default for UltraFace {
    fn default() -> Self {
        UltraFace {}
    }
}

impl ImageProcessor for UltraFace {
    fn process_image(
        &self,
        _image: &mut Box<DynamicImage>,
    ) -> Result<(), Box<dyn std::error::Error>> {
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

    #[test]
    fn default() {
        let ultraface = UltraFace::default();
        assert_eq!(ultraface.name(), "ultraface");
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
