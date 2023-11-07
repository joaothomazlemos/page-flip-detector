---
title: flipping-page-detector
emoji: 📖
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.1.1
app_file: src/models/app.py
pinned: false
---




page-flip-detector
==============================

# Data Description:

We collected page-flipping videos from smartphones and labeled them as flipping and not flipping.

We clipped the videos as short videos and labeled them as flipping or not flipping. The extracted frames are then saved to disk in sequential order with the following naming structure: VideoID_FrameNumber

# Goal(s):

Predict if the page is being flipped using a single image.

Success Metrics:

Evaluate model performance based on F1 score, the higher the better.

# Project Development

# Highlights

* Created a custom earlystopping class to save the best checkpoints weitghs states.
* Created custom model with nn class from Pytorch
* Used transfer learning technique with the light model of mobile Net v2
* Achieved best results with the custom model over the mobile net

# Image Classification with PyTorch

This Jupyter Notebook contains the code for training and evaluating image classification models using PyTorch. 

## Development

The notebook starts by importing the necessary libraries and loading the dataset. The dataset consists of images of pages being flipped or not, which are split into training and validation sets. The notebook then defines and trains two different models: cnn_model and mobilenet_v2. The first is built from scratch using the Pytorch nn module. The Mobile Net is a well-known mobile and light model, and we apply transfer learning to it.  After training both models on our dataset, we found that cnn_model performed better than mobilenet_v2, achieving an F1 score of 97.5%. This indicates that cnn_model is a good candidate for further testing and deployment.

## Conclusion


In this phase of testing, we trained and evaluated three different models: cnn_model, MobileNet, and ResNet. After training all three models on our dataset, we found that cnn_model performed the best, achieving an F1 score of 97.5%. However, MobileNet and ResNet also performed well, achieving F1 scores of 96.6% and 91.8%, respectively. 

These results indicate that all three models are good candidates for further testing and deployment. However, the task wanted the model to be applied on mobile applications, which often means that the model has to be smaller then 40 MB.

* Our custom CNN model got Estimated Total Size (MB): 51.09;

* Although ResNet18 is a popular and well performing model, it is not the best choice for mobile applications. ResNet18 got Estimated Total Size (MB): 81.11;

* MobileNetV2 is our choice: it is a small and efficient model that is well suited for mobile applications. MobileNetV2 got Estimated Total Size (MB): 24.88.

For future work, I intend to tweak on the custom model precision point using quantization techniques to reduze its size and try to fit in mobile applications.


![output](https://github.com/joaothomazlemos/page-flip-detector/assets/62029505/3159cadb-0185-4b0d-9443-5a0601199e6d)






Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
