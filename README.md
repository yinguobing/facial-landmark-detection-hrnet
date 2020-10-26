# facial-landmark-detection-hrnet
A TensorFlow implementation of HRNet for facial landmark detection.

![ms_marvel](./doc/../docs/ms_marvel.gif)

Watch this demo video: [HRNet Facial Landmark Detection (bilibili)](https://www.bilibili.com/video/BV1Vy4y1C79p/).

## Features
 - Support multiple public dataset: WFLW, IBUG, etc.
 - Advanced model architecture: HRNet v2
 - Data augmentation: randomly scale/rotate/flip
 - Model optimization: quantization, pruning

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.3-brightgreen)
![TensorFlow Model Optimizer Toolkit](https://img.shields.io/badge/TensorFlow_Model_Optimization_Toolkit-v0.5-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.3-brightgreen)
![Numpy](https://img.shields.io/badge/Numpy-v1.17-brightgreen)

### Installing
#### Get the source code for training

```shell
# From your favorite development directory
git clone --recursive https://github.com/yinguobing/facial-landmark-detection-hrnet.git
```

#### Generate the training data
There are multiple public facial mark datasets available which can be used to generate training heatmaps we need. For this training process the images will be augmented. The first step is transforming  the dataset into a more uniform distribution that is easier to process. You can do this yourself or, use this repo:

```shell
# From your favorite development directory
git clone https://github.com/yinguobing/face-mesh-generator.git

# Checkout the desired branch
git checkout features/export_dataset
```
Use the module `generate_mesh_dataset.py` to generate training data. Popular public datasets like IBUG, 300-W, WFLW are supported. Checkout the full list here: [facial-landmark-dataset](https://github.com/yinguobing/facial-landmark-dataset).


## Training

### Set the training and validation datasets

These files do not change frequently so set them in the source code. Take WFLW as an example.

```python
# In module `train.py`
# Training data.
train_files_dir = "/path/to/wflw_train"

# Validation data.
test_files_dir = "/path/to/wflw_test"
```

### Construct the model
The HRNet architecture is flexible. Custom the model if needed.

```python
model = hrnet_v2(width=18, output_channels=98)
```

`output_channels` equals to the number of facial marks of the dataset.

### Start training
Set the hyper parameters in the command line.

```Shell
python3 train.py --epochs=80 --batch_size=32
```

Training checkpoints can be found in directory `./checkpoints`. Before training started, this directory will be checked and the model will be restored if any checkpoint is available. Only the best model (smallest validation loss) will be saved.

### Monitor the training process
Use TensorBoard. The log and profiling files are in directory `./logs`

```shell
tensorboard --logdir /path/to/facial-landmark-detection-hrnet/log

```
## Evaluation
A quick evaluation on validation datasets will be performed automatically after training. For a full evaluation, please run the `evaluate.py` file. The NME value will be printed after evaluation.

```
python3 evaluate.py
```

## Export for inference
Exported model will be saved in `saved_model` format in directory `./exported`.

```shell
python3 train.py --export_only=True
```

## Inference
Check out module `predict.py` for details. If you want to test with video files or webcams, check out branch `features/with-face-detector`.

## Optimization
Optimize the model so it can run on mobile, embedded, and IoT devices. TensorFlow supports post-training quantization, quantization aware training, pruning, and clustering.

### Post training quantization
There are multiple means for post training quantization: dynamic range, integer only, float16. To quantize the model, run:

```bash
python3 quantization.py
```
Quantized tflite file will be find in the `optimized` directory.

### Pruning
Model pruning could dramatically reduce the model size while minimize the side effects on model accuracy. There is a demo video showing the performance of a pruned model with 80% of weights pruned (set to zero): [TensorFlow model pruning (bilibili)](https://www.bilibili.com/video/BV1Uz4y1o7Fb/)

To prune the model in this repo, run:

```bash
python3 pruning.py

```
Pruned model file will be find in the `optimized` directory.


### Quantization aware training
Due to the conflict between pruning and quantization aware training, please checkout the other branch for details.

```bash
git checkout features/quantization-aware-training
python3 train.py --quantization=True
```

## Authors
Yin Guobing (尹国冰) - yinguobing

![wechat](docs/wechat.png)

## License
![GitHub](https://img.shields.io/github/license/yinguobing/facial-landmark-detection-hrnet)

## Acknowledgments
The HRNet authors and the dataset authors who made their work public.
