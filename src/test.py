from PIL import Image
from numpy import asarray
import lenna_ultraface_plugin
print(lenna_ultraface_plugin.description())

image = Image.open('assets/lenna.png')
data = asarray(image)
print(data.shape)

config = lenna_ultraface_plugin.default_config()
print(config)
processed = lenna_ultraface_plugin.process(config, data)
print(processed.shape)
Image.fromarray(processed).save('lenna_test_out.png')
