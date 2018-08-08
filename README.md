# PredictMovingDirection
This repository contains the code for performing the experiments in the paper: "Predicting the future direction of cell movement with convolutional neural networks".
This project is carried out by [Funahashi lab at Keio
University](https://fun.bio.keio.ac.jp).

The commands below assume starting from the "PredictMovingDirection".

REQUIREMENTS
-------------
* [Python 3.6+](https://www.python.org/downloads/)
* [CUDA 6.5, 7.0, 7.5, 8.0](https://developer.nvidia.com/cuda-zone) (for GPU support)
* [cuDNN v2, v3, v4, v5, v5.1, v6](https://developer.nvidia.com/cudnn) (for GPU support)
* [Chainer 1.24+](https://chainer.org/)
* [NumPy](http://www.numpy.org/)
* [opencv-python](https://pypi.org/project/opencv-python/)
* [pandas](https://pandas.pydata.org/getpandas.html)
* [scikit-image](https://scikit-image.org/)

You can use the following command to install the above Python packages and dependencies:

```sh
pip install -r requirements.txt
```


GETTING THE NIH/3T3 DATASET
----------------------------
The NIH/3T3 image dataset for training and validation of the CNN models are too large to store in the repository. Please download the dataset using instructions below.

To download and unzip the dataset, please run:

```sh
./download_dataset.sh
```

Or download the dataset directly from [here](https://fun.bio.keio.ac.jp/software/MDPredictor/NIH3T3_4foldcv.zip) (52MB) and unzip it.


PREPARING THE NIH/3T3 DATASET FROM THE RAW IMAGES
---------------------------------------------------
Instead of downloading the image dataset as described above, you can prepare the dataset from the raw images by the following procedure:

1. Download the raw time-lapse phase contrast images of NIH/3T3 fibroblasts

    To download and unzip, please run:

    ```sh
    ./download_raw.sh
    ```

    Otherwise, please download it from
    [here](https://fun.bio.keio.ac.jp/software/MDPredictor/NIH3T3_timelapse.zip) (1.2GB), and unzip it.

2. Annotate moving direction and create image patches

    Please run:

    ```sh
    python ./src/prepare_dataset/annotate_NIH3T3.py --in_dir /path/to/raw_images
    ```

    Annotated image patches and their motility measure will be saved in `./NIH3T3_annotated` directory.

3. Create dataset for training and validation of the CNN models

    Please run:

    ```sh
    python ./src/prepare_dataset/make_cv_data.py
    ```

    The NIH/3T3 image dataset will be saved in `./NIH3T3_4foldcv` directory.


TRAINING AND VALIDATION OF CNN MODELS
--------------------------------------
To train and test CNN models in 4-fold cross-validation, run:

```sh
./cross_val.sh -d path/to/dataset -r path/to/results [-g GPU_id]
```

  * specify the argument of `-d`, which indicates the dataset directory for training and validation of the CNN models (e.g., downloaded `NIH3T3_4foldcv`)
  * specify the argument of `-r`, which indicates the results directory where resulting models and training results will be saved
  * the argument passed to `-g` indicates id of GPU used for computation (negative value indicates CPU; default is -1)

Results directory will have the following structure:
```
results/
  +-- summary.json (Result of 4-fold cross validation)
  +-- fold0/
  |       +-- best_score.json (Best score of each evaluation criterion)
  |       +-- log (Training log)
  |       +-- model.npz (Trained model)
  |       +-- train_detail.json  (Configuration of training)
  +-- fold1/
          +-- ...
```


VISUALIZING LOCAL IMAGE FEATURES LEARNED BY THE CNN MODELS
---------------------------------------------------------
The local feature that most strongly activates a CNN's particular neuron can be visualized by using guided backpropagation (GBP)[[1]](#ref1). To visualize the local features for the feature maps of the last convolutional layers, please run:

```sh
./visualization_gbp.sh -d path/to/dataset -r path/to/results [-g GPU_id]
```

  * assumes that you have done 4-fold cross validation as described above
  * specify the argument of `-d`, which indicates the dataset directory used for 4-fold cross validation
  * specify the argument of `-r`, which indicates the results directory where the 4-fold cross validation results are saved

The visualization results will be saved in the results directory with the following structure:

```
results/
    +-- fold0/
    |   +-- gbp/ (Results of visualization by using GBP)
    |       +-- config.json (Configuration of visualization)
    |       +-- correct/ (Results for correctly predicted test images)
    |       |   +-- {moving direction}/ (Annotated moving direction)
    |       |       +-- {cell index}/ (Index to distinguish each image patch of cell)
    |       |           +-- local_feature_rank0_ch{channel index}.tif
    |       |           |   (Local feature for the feature map with the first highest activation)
    |       |           +-- local_feature_rank1_ch{channel index}.tif
    |       |           |   (Local feature for the feature map with the second highest activation)
    |       |           +-- local_feature_rank1_ch{channel index}.tif
    |       |           |   (Local feature for the feature map with the third highest activation)
    |       |           +-- pred_result.txt (Prediction result by the trained model)
    |       +-- incorrect/ (Results for incorrectly predicted test images)
    |           +-- ...
    +-- fold1/
    ...
```


VISUALIZING GLOABL IMAGE FEATURES LEARNED BY THE CNN MODELS
------------------------------------------------------------
The degree of contribution to the CNN prediction can be calculated for each pixel of the input image by using deep Taylor decomposition (DTD)[[2]](#ref2). Calculated pixel-wise contribution can be visualized as a heatmap. To visualize pixel-wise contribution to CNN prediction results, please run:

```sh
./visualization_dtd.sh -d path/to/dataset -r path/to/results [-g GPU_id]
```

  * assumes that you have done 4-fold cross validation as described above
  * specify the argument of `-d`, which indicates the dataset directory used for 4-fold cross validation
  * specify the argument of `-r`, which indicates the results directory where the 4-fold cross validation results are saved

The visualization results will be saved in the results directory with the following structure:

```
results/
    +-- fold0/
    |   +-- dtd/ (Results of visualization by using DTD)
    |       +-- config.json (Configuration of visualization)
    |       +-- correct/ (Results for correctly predicted test images)
    |       |   +-- {moving direction}/ (Annotated moving direction)
    |       |       +-- {cell index}/ (Index to distinguish each image patch of cell)
    |       |           +-- rel_heatmap.tif (Heatmap of pixel-wise contribution)
    |       |           +-- relevance.npy (Calculated pixel-wise contribution)
    |       |           +-- pred_result.txt (Prediction result by the trained model)
    |       +-- incorrect/ (Results for incorrectly predicted test images)
    |           +-- ...
    +-- fold1/
    ...
```


REFERENCES
-----------------------
<a name="ref1"></a> [[1] Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.](https://arxiv.org/abs/1412.6806)   
<a name="ref2"></a> [[2] Gr´egoire Montavon, Sebastian Lapuschkin, Alexander Binder, Wojciech Samek, and Klaus-Robert Müller. Explaining nonlinear classification decisions with deep Taylor decomposition. Pattern Recognition, 65:211–222, 2017.](https://www.sciencedirect.com/science/article/pii/S0031320316303582)
