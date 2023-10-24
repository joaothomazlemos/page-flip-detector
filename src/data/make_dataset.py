# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob
import numpy as np
import torch.utils.data
import torchvision.transforms as T
import torchvision
import os


def train_transform():
    """
    Data augmentation for training data"""
    train_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=np.array([0.57647944, 0.52539918, 0.49818376]),
                 std=np.array([0.21990674, 0.23702262, 0.24528695]))
    ])
    return train_transform

def test_transform():
    """
    Data augmentation for test data"""
    test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=np.array([0.57647944, 0.52539918, 0.49818376]),
                 std=np.array([0.21990674, 0.23702262, 0.24528695]))
    ])
    return test_transform
   

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    #set the data paths
    DATA_DIR = input_filepath
    DATA_DIR_RAW = os.path.join(DATA_DIR, 'raw')
    DATA_DIR_RAW_IMG = os.path.join(DATA_DIR_RAW, 'images')
    DATA_DIR_RAW_IMG_TRAIN = os.path.join(DATA_DIR_RAW_IMG, 'training')
    DATA_DIR_RAW_IMG_TEST = os.path.join(DATA_DIR_RAW_IMG, 'testing')

    #creating the datasets , with imagefolder to load the images even if they are in different folders. v1
    train_dataset = torchvision.datasets.ImageFolder(root=DATA_DIR_RAW_IMG_TRAIN,
                                                    transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=DATA_DIR_RAW_IMG_TEST,
                                                        transform=test_transform)
    #saving the datasets
    torch.save(train_dataset, os.path.join(output_filepath, 'train_dataset.pt'))
    torch.save(test_dataset, os.path.join(output_filepath, 'test_dataset.pt'))

    
    






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    #often useful for finding various files when not using click
    #current_file_path = Path(__file__).resolve()
    #
    #project_dir = current_file_path.parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main() # loads data from raw and saves here on src/data


#to run the script:
#python src/data/make_dataset.py

# the input folder is the data folder
#data

#the folder of this script is src/data, is the folder we save the DS
#src/data

#we run like:
#python src/data/make_dataset.py data src/data
