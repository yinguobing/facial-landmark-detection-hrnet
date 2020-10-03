# facial-landmark-detection-hrnet
A TensorFlow implementation of HRNet for facial landmark detection.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
For training:

![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.3-brightgreen)

For inference:

![OpenCV](https://img.shields.io/badge/OpenCV-v4.3-brightgreen)
![Numpy](https://img.shields.io/badge/Numpy-v1.17-brightgreen)

### Installing
#### Get the source code for training.

```shell
# From your favorite development directory
git clone https://github.com/yinguobing/facial-landmark-detection-hrnet.git
```

#### Generate the training data.
There are multiple public facial mark datasets available which can be used to generate training heatmaps we need. You can do this yourself or, use this repo:

```shell
# From your favorite development directory
git clone https://github.com/yinguobing/face-mesh-generator.git
```
Use the module `generate_heatmap_dataset.py` to generate training data. Popular public datasets like IBUG, 300-W, WFLW are supported. Checkout the full list here: [facial-landmark-dataset](https://github.com/yinguobing/facial-landmark-dataset).


## Training

### Set the training and validation datasets. 

These files do not change frequently so set them in the source code. Take WFLW as an example.

```python
# In module `train.py`
# Training data.
record_file_train = "/path/to/wflw_train.record"

# Validation data.
record_file_test = "/path/to/wflw_test.record"
```

Also make sure the image size and heatmap size are in accordance with the dataset.

```python
# In _parse_function()
image_decoded = tf.reshape(image_decoded, [256, 256, 3])
heatmaps = tf.reshape(heatmaps, (98, 64, 64))
```
### Construct the model.
The HRNet architecture is flexible. Custom the model if needed.

```python
model = HRNetV2(width=18, output_channels=98)
```

`output_channels` equals to the number of facial marks of the dataset.

### Start training.
Set the hyper parameters in the command line.

```Shell
python3 train.py --epochs=80 --batch_size=32
```

Training checkpoints can be found in directory `./checkpoints`. Before training started, this directory will be checked and the model will be restored if any checkpoint is available. Only the best model (smallest validation loss) will be saved.

### Monitor the training process
Use TensorBoard. The log and profiling files are in directory `./log`

```shell
tensorboard --logdir /path/to//facial-landmark-detection-hrnet/log

```
## Evaluation
Evaluation on validation datasets will be performed automatically after training. But you can perform evaluation without training like this:

```
python3 train.py --eval_only=True
```
Do not forget setting the validation dataset.

## Export for inference
Exported model will be saved in `saved_model` format in directory `./exported`.
```shell
python3 train.py --export_only=True
```

## Inference
Check out module `predict.py` for details.

## Authors
Yin Guobing (尹国冰) - yinguobing

![wechat](docs/wechat.png)

## License
![GitHub](https://img.shields.io/github/license/yinguobing/facial-landmark-detection-hrnet)

## Acknowledgments
The HRNet authors and the dataset authors who made their work public.
