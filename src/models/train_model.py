#---------Importing libraries---------#
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchinfo import summary
from tqdm import tqdm  # Progress bar for py scripts
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
from skimage import io
#importing our custom class EarlyStop that is in the utils folder
from src.utilities.earlystop import EarlyStopping
from src.data.make_dataset import MakeDataset
import torch


def train_data_loader(train_dataset, batch_size=16):
    """
    Data loader for training data
    
    Args:
    train_dataset: PyTorch dataset object containing the training data
    batch_size: int, batch size for the data loader
    
    Returns:
    train_loader: PyTorch data loader object for the training data
    """
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)
    return train_loader

def test_data_loader(test_dataset):
    """
    Data loader for test data
    
    Args:
    test_dataset: PyTorch dataset object containing test data
    
    Returns:
    PyTorch DataLoader object for test data
    """
    return torch.utils.data.DataLoader(dataset=test_dataset,
                                        batch_size=16,
                                        shuffle=False
                                        )

def train_model(model_name, model, criterion, optimizer, trainloader, valloader, epochs=5, patience=4, verbose=True):
    """
    Training loop for the model. It saves the best model state dict
    
    _________________________________________________________________
    Parameters:
    model_name: name of the model
    model: model to be trained
    criterion: loss function
    optimizer: optimizer
    trainloader: train data loader
    valloader: validation data loader
    epochs: number of epochs
    patience: number of epochs without improvement before stopping the training
    verbose: if True, prints the results of each epoch
    model_dir: directory where the model will be saved
    _________________________________________________________________
    Returns:
    results: dict with the results of the training
    train_loss: list with the training loss of each epoch
    train_accuracy: list with the training accuracy of each epoch
    val_loss: list with the validation loss of each epoch
    val_accuracy: list with the validation accuracy of each epoch
    """


    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    #checks if the model dir folder exists, if not it creates it
    if not os.path.exists('src/data/models/'+model_name):
        os.makedirs('src/data/models/'+model_name)
    #initialize the last best weitghts
    if os.path.exists('src/data/models/'+model_name+'/checkpoint.pt'):
        model.load_state_dict(torch.load('src/data/models/'+model_name+'/checkpoint.pt'),   torch.device(device))
        print('Loaded checkpoint with the best model.')
    # initialize the early_stopping object
    early_stopping = EarlyStopping(model_name=model_name, patience=patience, verbose=True)

    #using tqdm to show the progress bar

    for epoch in tqdm(range(epochs)):
        train_batch_loss = 0
        train_batch_acc = 0
        val_batch_loss = 0
        val_batch_accuracy = 0
        

        # Training mode
        model.train()
        print('Training model on the epoch batches...')
        for X, y in tqdm(trainloader):
            ## move data to device (GPU if available)
            if device != "cpu":
                X = X.to(device)
                y = y.to(device)
            ## reset the gradient
            optimizer.zero_grad()
            ## forward pass
            y_hat = model(X).flatten()
            ## calculate the loss
            loss = criterion(y_hat, y.type(torch.float32))
            ## backpropagation
            loss.backward()
            ## update the weights
            optimizer.step()
            train_batch_loss += loss.item()
            train_batch_acc += (torch.round(y_hat) == y).type(torch.float32).mean().item()
        train_loss.append(train_batch_loss / len(trainloader))
        train_accuracy.append(train_batch_acc / len(trainloader))
        
        
        # Validation
        model.eval()
        print('Validating model on epoch..')
        for data, target in tqdm(valloader):
            
            # move tensors to GPU if CUDA is available
            if device != "cpu":
                data, target = data.to(device), target.to(device)
            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data).flatten()
            loss = criterion(output, target.type(torch.float32))
            val_batch_loss += loss.item()
            val_batch_accuracy += (torch.round(output) == target).type(torch.float32).mean().item()
            

        ## record validation loss
        val_loss.append(val_batch_loss / len(valloader))
        val_accuracy.append(val_batch_accuracy / len(valloader))
        

        ## Print progress
        if verbose:
            print(f"Epoch {epoch + 1}:",
                  f"Train Loss: {train_loss[-1]:.3f}",
                  f"Train Accuracy: {train_accuracy[-1]:.2f}",
                  f"Validation Loss: {val_loss[-1]:.3f}",
                  f"Validation Accuracy: {val_accuracy[-1]:.2f}")
            
        
        ## early_stopping needs the validation loss to check if it has decresed, 
        ## and if it has, it will make a checkpoint of the current model
        val_loss_arr = np.average(val_loss)
        early_stopping(val_loss_arr, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

       
    results = {"train_loss": train_loss,
               "train_accuracy": train_accuracy,
               "val_loss": val_loss,
               "val_accuracy": val_accuracy}
    
        # load the last checkpoint with the best model
    model.load_state_dict(torch.load('src/data/models/'+model_name+'/checkpoint.pt'), torch.device(device))
    #save the model dict state
    torch.save(model.state_dict(), 'src/data/models/'+model_name+'/model.pt')
   

    return results, train_loss, train_accuracy, val_loss, val_accuracy

#saving results
def save_model_results(train_loss, train_accuracy, val_loss, val_accuracy, model_name:str):
    """
    This function saves the results of the model training and validation in the results folder
    So it can be used later to visuallize how the training was

    _________________________________________________________________
    Parameters:
    train_loss: list with the training loss of each epoch
    train_accuracy: list with the training accuracy of each epoch
    val_loss: list with the validation loss of each epoch
    val_accuracy: list with the validation accuracy of each epoch
    model_name: name of the model
    _________________________________________________________________
    Returns:
    None
    
    """
    #creating a folder inside results dir. The folder name is the model name
    #first check if the folder exists
    results_dir = 'src/data/results'
    if not os.path.exists(os.path.join(results_dir, model_name)):
        os.makedirs(os.path.join(results_dir, model_name))
    #save the results
    #save the training loss and accuracy
    np.save(os.path.join(results_dir, model_name, 'train_loss.npy'), np.array(train_loss))#
    np.save(os.path.join(results_dir, model_name, 'train_loss.npy'), np.array(train_loss))
    np.save(os.path.join(results_dir, model_name, 'train_accuracy.npy'), np.array(train_accuracy))
    # save the validation loss and accuracy
    np.save(os.path.join(results_dir, model_name, 'val_loss.npy'), np.array(val_loss))
    np.save(os.path.join(results_dir, model_name, 'val_accuracy.npy'), np.array(val_accuracy))
    print('Results saved!')

#loading the model we want to use here
def load_model_configs():
    """
    Loading the model, the model optimizer and the loss function for that especif model.
    As we are using transfer learning, we are loading the model from pytorch hub and
    changing the last layer to fit our problem"""

    #loading the model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT')


    #freezing the parameters for the  layers
    for param in model.parameters():
        param.requires_grad = False

    #changing the last  layer
    model.classifier[1] = nn.Linear(1280, 1)
    #adding sigmoid to the output
    model.classifier.add_module('sigmoid', nn.Sigmoid())
    #optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #loss for this model
    loss_fn = nn.BCEWithLogitsLoss() #binary cross entropy loss with logits (sigmoid is not applied to the output of the model)
    #loss_fn = nn.BCELoss() #binary cross entropy loss without logits (sigmoid is applied to the output of the model)

    return model, optimizer, loss_fn



def main():

    torch.manual_seed(42)  # Setting the seed

    #loading the processed data
    data_filepath = 'data'

    train_dataset, test_dataset = MakeDataset(data_filepath).process_data()
    print('Data loaded and processed!')


    if torch.cuda.is_available(): # GPU operations have a separate seed we also want to set
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

    #loading model and model configs
    model, optimizer, loss_fn = load_model_configs()
    print('Model loaded!')

    

    _, train_loss, train_accuracy, val_loss, val_accuracy = train_model(model.__class__.__name__,
                                                                               model,
                                                                                 criterion=loss_fn,
                                                                                   optimizer=optimizer,
                                                                                     trainloader=train_data_loader(train_dataset),
                                                                                       valloader=test_data_loader(test_dataset),
                                                                                         epochs=1,
                                                                                           patience=3,
                                                                                             verbose=True)
    
    

    save_model_results(train_loss, train_accuracy, val_loss, val_accuracy, model.__class__.__name__)

    #saving the model arvhiteture
    torch.save(model, 'src/data/models/'+model.__class__.__name__+'/model.pth')



if __name__ == "__main__":
    main()