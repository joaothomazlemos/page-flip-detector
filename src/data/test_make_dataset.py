import os
import unittest
import torch
import torchvision.transforms as T
from src.data.make_dataset import make_dataset


class TestMakeDataset(unittest.TestCase):
    
    def setUp(self):
        self.make_dataset = make_dataset()
        self.data_filepath = 'data'
        
    def test_train_transform(self):
        train_transform = self.make_dataset.train_transform()
        self.assertIsInstance(train_transform, T.Compose)
        
    def test_test_transform(self):
        test_transform = self.make_dataset.test_transform()
        self.assertIsInstance(test_transform, T.Compose)
        
    def test_process_data(self):
        train_dataset, test_dataset = self.make_dataset.process_data(self.data_filepath)
        self.assertIsInstance(train_dataset, torch.utils.data.Dataset)
        self.assertIsInstance(test_dataset, torch.utils.data.Dataset)

    def test_process_data_filepath(self):
        self.assertEqual('data', self.data_filepath)
        
if __name__ == '__main__':
    unittest.main()