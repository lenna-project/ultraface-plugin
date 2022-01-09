# ultraface-plugin
Face Detection in images.
Based on [ultraface](https://github.com/onnx/models/tree/master/vision/body_analysis/ultraface) model.

Test it on [lenna.app](https://lenna.app/?just=https://lenna.app/ultraface-plugin/remoteEntry.js).

## rust version

The plugin is nativaly developed in rust programming language.

### build

Build the plugin.

```bash
cargo build --release
```

The file target/release/liblenna_ultraface_plugin.so can be copied to the plugins folder of

[lenna-cli](https://github.com/lenna-project/lenna-cli) and used in the pipeline.

## wasm and javascript version

The plugin can be compiled to wasm and used on [lenna.app](https://lenna.app).

### build

Build the wasm package.

```bash
wasm-pack build
```

The node module can be build then.

```bash
npm run build
```

### serve

The plugin can be hosted using

```bash
npm run start
```

A server runs on localhost:3002 now.

On [lenna.app](https://lenna.app) on the left side the url http://localhost:3002 can be loaded to use the plugin.

## python version

The plugin can build python bindings.

### build

Create a virtual environment for python.

```bash
virtualenv -p python3 .venv
source .venv/bin/activate
pip install .
```

### usage

Import lenna_ultraface_plugin in a python environment.

```python
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
```

### test

Run the python [test file](src/test.py) which loads the [lenna.png](assets/lenna.png) and converts it.

```bash
pip install pillow numpy
python src/test.py
```

### jupyter notebook

Find an example in [example.ipynb](example.ipynb)

```bash
pip install jupyter
jupyter notebook example.ipynb
```
