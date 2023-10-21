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

In this phase of testing, we trained and evaluated two different models: cnn_model and mobilenet_v2. After training both models on our dataset, we found that cnn_model performed better than mobilenet_v2, achieving an F1 score of 97.5%. This indicates that cnn_model is a good candidate for further testing and deployment. However, it's important to note that this F1 score was achieved on a specific dataset and may not generalize well to other datasets. Therefore, further testing and evaluation are necessary before deploying this model in a production environment.

## Results

The best model was the 'cnn_model' with Train Loss: 0.037 Train Accuracy: 0.99 Validation Loss: 0.021 Validation Accuracy: 0.99. We also made predictions on the test set and computed the F1 score, which was 97.5%.

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
