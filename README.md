# PredictMovingDirection
This repository contains the code for performing the experiments in the paper: ["Predicting the future direction of cell movement with convolutional neural networks"](https://doi.org/10.1101/388033).
This project is carried out by [Funahashi lab at Keio
University](https://fun.bio.keio.ac.jp).

The commands below assume starting from the `PredictMovingDirection/` directory.

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
* [scikit-learn](http://scikit-learn.org/stable/)

You can use the following command to install the above Python packages and dependencies:

```sh
pip install -r requirements.txt
```


GETTING THE IMAGE DATASETS
----------------------------
The image datasets for training and validation of the CNN models are too large to store in the repository. Please download the datasets using instructions below.

To download and unzip the datasets, please run:

```sh
./download_datasets.sh
```

Or download the datasets directly from [here](https://fun.bio.keio.ac.jp/software/MDPredictor/datasets.zip) (85.5MB) and unzip it.

Downloaded datasets directory (`datasets`) contains the NIH/3T3 dataset (`NIH3T3_4foldcv`) and the U373 dataset (`U373_dataset`).

PREPARING THE DATASETS FROM THE RAW IMAGES
---------------------------------------------------
Instead of downloading the image datasets as described above, you can prepare the datasets from the raw images by the following procedure:

1. Download the raw time-lapse phase contrast images

    To download and unzip, please run:

    ```sh
    ./download_raw.sh
    ```

    Otherwise, please download it from
    [here](https://fun.bio.keio.ac.jp/software/MDPredictor/raw_images.zip) (1.25GB), and unzip it.

    Downloaded raw images directory (`raw_images`) contains the raw time-lapse phase contrast images and manual tracking results of NIH/3T3 (`NIH3T3_timelapse`) and U373 (`U373_timelapse`) cells. Images of U373 cells are the part of the dataset used in the ISBI (International Symposium on Biomedical Imaging) cell tracking challenge 2015[[1]](#ref1)[[2]](#ref2).

2. Annotate moving direction and create image patches

    For images of NIH/3T3, please run:

    ```sh
    python ./src/prepare_dataset/annotate_NIH3T3.py --in_dir /path/to/raw_images/NIH3T3_timelapse
    ```

    For images of U373, please run:

    ```sh
    python ./src/prepare_dataset/annotate_U373.py --in_dir /path/to/raw_images/U373_timelapse
    ```

    Annotated image patches and their motility measure will be saved in `./NIH3T3_annotated` or `./U373_annotated`, respectively.

3. Create dataset for training and validation of the CNN models

    Please run:

    ```sh
    python ./src/prepare_dataset/make_cv_data.py --in_dir /path/to/annotated_images --out_dir /path/to/dataset
    ```

    * specify the argument of `--in_dir`, which indicates the annotated image directory (e.g., `./NIH3T3_annotated`)
    * specify the argument of `--out_dir`, which indicates the dataset directory where created dataset will be saved (e.g., `./datasets/NIH3T3_4foldcv`)


TRAINING AND VALIDATION OF CNN MODELS
--------------------------------------
To train and test CNN models in 4-fold cross-validation, run:

```sh
./cross_val.sh -d path/to/dataset -r path/to/results [-g GPU_id]
```

  * specify the argument of `-d`, which indicates the dataset directory for training and validation of the CNN models (e.g., downloaded `./datasets/NIH3T3_4foldcv`)
  * specify the argument of `-r`, which indicates the results directory where resulting models and training results will be saved (e.g., `./NIH3T3_results`)
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
The local feature that most strongly activates a CNN's particular neuron can be visualized by using guided backpropagation (GBP)[[3]](#ref3). To visualize the local features for the feature maps of the last convolutional layers, please run:

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
The degree of contribution to the CNN prediction can be calculated for each pixel of the input image by using deep Taylor decomposition (DTD)[[4]](#ref4). Calculated pixel-wise contribution can be visualized as a heatmap. To visualize pixel-wise contribution to CNN prediction results, please run:

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
<a name="ref1"></a> [[1] Martin Maˇska, Vladim ́ır Ulman, David Svoboda, Pavel Matula, Petr Matula, Cristina Ederra, Ainhoa Urbiola, Tom ́as Espan ̃a, Subramanian Venkatesan, Deepak MW Balak, et al. A benchmark for comparison of cell tracking algorithms. Bioinformatics, 30(11):1609–1617, 2014.](https://academic.oup.com/bioinformatics/article/30/11/1609/283435)   
<a name="ref2"></a> [[2] Vladim ́ır Ulman, Martin Maˇska, Klas EG Magnusson, Olaf Ronneberger, Carsten Haubold, Nathalie Harder, Pavel Matula, Petr Matula, David Svo- boda, Miroslav Radojevic, et al. An objective comparison of cell-tracking algorithms. Nature methods, 14(12):1141, 2017.](https://www.nature.com/articles/nmeth.4473)   
<a name="ref3"></a> [[3] Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.](https://arxiv.org/abs/1412.6806)   
<a name="ref4"></a> [[4] Gr´egoire Montavon, Sebastian Lapuschkin, Alexander Binder, Wojciech Samek, and Klaus-Robert Müller. Explaining nonlinear classification decisions with deep Taylor decomposition. Pattern Recognition, 65:211–222, 2017.](https://www.sciencedirect.com/science/article/pii/S0031320316303582)
