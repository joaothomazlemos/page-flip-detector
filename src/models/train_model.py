#---------Importing libraries---------#
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.datasets
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchinfo import summary
from tqdm.notebook import tqdm  # Progress bar
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import pandas as pd
from skimage import io
#importing our custom class EarlyStop that is in the utils folder
from src.utils.EarlyStop import EarlyStopping



#creating our dataloader function

def train_data_loader(train_dataset, batch_size=16):
    """
    Data loader for training data"""
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)
    return train_loader

def test_data_loader(test_dataset):
    """
    Data loader for test data"""
    return torch.utils.data.DataLoader(dataset=test_dataset)

def train_model(model_name, model, criterion, optimizer, trainloader, valloader, epochs=5, patience=7, verbose=True, model_dir=MODEL_DIR):
    
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    #checks if the model results folder exists, if not it creates it
    if not os.path.exists(os.path.join(model_dir, model_name)):
        os.makedirs(os.path.join(model_dir, model_name))
    #initialize the last best weitghts
    if os.path.exists(os.path.join(model_dir, model_name,'checkpoint.pt')):
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name, 'checkpoint.pt'), map_location=torch.device(device)))
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
        print('Training model on the epoch...')
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
            sigmoid = nn.Sigmoid()
            y_hat = sigmoid(y_hat)
            loss = criterion(y_hat, y.type(torch.float16)) #BCEloss. Check the shapes of the tensors . Precison 16 to reduce memory usage
            ## backpropagation
            loss.backward()
            ## update the weights
            optimizer.step()
            train_batch_loss += loss.item()
            train_batch_acc += (torch.round(y_hat) == y).type(torch.float16).mean().item()
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
            # calculate the loss
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
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name, 'checkpoint.pt')))
   

    return results, train_loss, train_accuracy, val_loss, val_accuracy

#saving results
RESULTS_DIR = os.path.join(MODEL_DIR, 'results')
def save_model_results(train_loss, train_accuracy, val_loss, val_accuracy, model_name:str, results_dir = RESULTS_DIR ):
    #creating a folder inside results dir. The folder name is the model name
    #first check if the folder exists
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


def main():

    torch.manual_seed(42)  # Setting the seed

    # get device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if torch.cuda.is_available(): # GPU operations have a separate seed we also want to set
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #MODEL_DIR = os.path.join(project_path, 'models')

    #optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.BCELoss() #binary cross entropy loss

    results, train_loss, train_accuracy, val_loss, val_accuracy = train_model(model.__class__.__name__,
                                                                               model,
                                                                                 criterion=loss_fn,
                                                                                   optimizer=optimizer,
                                                                                     trainloader=train_loader()
                                                                                       valloader=test_loader(),
                                                                                         epochs=20,
                                                                                           patience=3,
                                                                                             verbose=True)
    


