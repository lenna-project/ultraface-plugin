const pkg = import("../pkg");
export const processor = pkg;
export const name = () => "ultraface";
export const description = () => "Plugin to label images.";
export const process = async (config, image) => {
  return import("../pkg").then((processor) => processor.process(config, image));
};
export const defaultConfig = async () => {
  return {};
};
