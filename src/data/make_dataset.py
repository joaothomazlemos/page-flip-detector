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

class MakeDataset():


    def __init__(self,data_filepath):
        self.data_filepath = data_filepath




    def train_transform(self):
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

    def test_transform(self):
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
    


    def process_data(self):
        """
        Runs data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed.
        
        Args:
        input_filepath (str): The path to the input directory containing the data directory (project/data)
        train_transform (torchvision.transforms): The transformation to be applied to the training dataset.
        test_transform (torchvision.transforms): The transformation to be applied to the testing dataset.
        
        Returns:
        train_dataset (torchvision.datasets.ImageFolder): The training dataset.
        test_dataset (torchvision.datasets.ImageFolder): The testing dataset.
        """
        #set the data paths
        DATA_DIR = self.data_filepath
        DATA_DIR_RAW = os.path.join(DATA_DIR, 'raw')
        DATA_DIR_RAW_IMG = os.path.join(DATA_DIR_RAW, 'images')
        DATA_DIR_RAW_IMG_TRAIN = os.path.join(DATA_DIR_RAW_IMG, 'training')
        DATA_DIR_RAW_IMG_TEST = os.path.join(DATA_DIR_RAW_IMG, 'testing')

        #creating the datasets , with imagefolder to load the images even if they are in different folders. v1
        train_dataset = torchvision.datasets.ImageFolder(root=DATA_DIR_RAW_IMG_TRAIN,
                                                        transform=self.train_transform())
        test_dataset = torchvision.datasets.ImageFolder(root=DATA_DIR_RAW_IMG_TEST,
                                                            transform=self.test_transform())
        #return the datasets

        return train_dataset, test_dataset

    