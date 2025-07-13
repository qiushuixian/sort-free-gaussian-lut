# Unofficial Implementation of Sort-Free Gaussian Splatting's Differentiable Gaussian Rasterization

Our `main.cu` function and the `CMakeLists.txt` file were written with reference to the following repository: [VladimirYugay/gaussian_rasterizer](https://github.com/VladimirYugay/gaussian_rasterizer).

This repository encompasses the differentiable Gaussian rasterization component of the sort-free Gaussian splatting technique.

Not only does it incorporate the rasterization segment callable from Python, consistent with the original usage, but it also furnishes test data and the `main.cu` file to facilitate the debugging of CUDA code.

We continue to use a tile-based rendering method same as 3DGS. The code still includes a Radix Sort, but we have removed the GS depth component from the keys required by the Radix Sort. This means that sorting is now based solely on the tile ID to which the GS belongs. This approach is a form of lazy approach, using sorting as a substitute for classification. While this change may impact performance, it will not affect rendering quality.

## Installation

### Python Integration

```shell
git clone https://github.com/LiYukeee/sort-free-diff-gaussian-rasterization.git
cd sort-free-diff-gaussian-rasterization
pip install .
```

### Debugging Setup

1. Install `glm` (already included in the `third_party` directory).

2. Install `LibTorch`.

Procure the appropriate version of LibTorch for your CUDA setup from https://pytorch.org/ as follows:

```shell
cd third_party
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu118.zip
```

3. Prepare Test Data
   
First, download the data from the specified source: [Google Drive](https://drive.google.com/drive/folders/1g0awLh2Ud7VDEXrQVfsbe5j9GAlJG2dZ?usp=sharing) and place it in the `test_data` folder.

All the test data was obtained from Replica dataset. Adjust the input paths and process the downloaded dump files with the command:

```shell
cd test_data
python convert_dump.py
```

This would unpack the dump files to .pt tensors.

4. Compilation and Execution

```shell
cd sort-free-diff-gaussian-rasterization
mkdir output
mkdir build
cd build
cmake ..
make
```

This would create rasterizer executable which can be examined with the VSCode debugger.

You can run it directly.

```shell
./rasterizer
```

## Examine the Results

You can create a file named xxx.ipynb in the directory and run the following code to display the image.

```python
from notebook_utils import *
image = loadTensorC("output/render.pt")
image = image.clamp(0, 1)
showImage(image)
```

## Debug and Release Configurations

We provide a `launch.json` file for Visual Studio Code to facilitate the debugging of CUDA code.

To toggle between Debug and Release modes, manually modify the `CMakeLists.txt` file.

Debug mode is intended for debugging purposes, whereas Release mode omits extraneous debugging information, making it suitable for recording precise timing metrics.

## Others

`notebook_utils.py` and `cuda_rasterizer/debug_utils.h` can help you conveniently debug the code.