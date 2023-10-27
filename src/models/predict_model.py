#import necessary librarys for image prediction

import torch
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
#image library
from PIL import Image
#import loader
from torch.utils.data import DataLoader
import glob
import os




def test_transform(img):
    """
    Data augmentation for test data"""
    transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=np.array([0.57647944, 0.52539918, 0.49818376]),
                std=np.array([0.21990674, 0.23702262, 0.24528695]))
    ])
    return transform(img)

def get_device():
    """
    Function to get device information
    
    Returns:
    Device information
    """
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    return device

def threshold_prediction(prediction):
        if prediction > 0.5:
            prediction = 'Not-Flipping'
        else:
            prediction = 'Flipping'
            return prediction



def main(image_folder, single_image=False, image_name=None):
    """
    Predicts whether a book page is being flipped or not, based on an input image.

    Args:
        image_path (str): The path to the input image.

    Returns:
        None
    """
    #set device avaible
    device = get_device()
 
    model = torch.load('src/data/models/MobileNetV2/model.pth')
    #load dictionary of model state weitghts
    model.load_state_dict(torch.load('src/data/models/MobileNetV2/checkpoint.pt'))
    #set model to evaluation mode
    model.eval()

    #set model to device
    model.to(device)
    #getting the class of the pred

    
    #plot the image with predicted label
    #0 = flipping the page
    #1 = not flipping the page

    if single_image == True:
        img = Image.open(os.path.join(image_folder, image_name))
        #transform image
        input = test_transform(img)
        input = torch.unsqueeze(input, 0)
        #predict
        prediction = model(input.to(device))
        prediction = threshold_prediction(prediction)
        plt.title('Prediction: {}'.format(prediction))
        plt.imshow(img)
        plt.show()
    else:

    #create a grid for the predictions for all the images in the example folder if it is the images in a folder to predict
        for i in range(0, len(glob.glob('src/data/examples/*.jpg'))):
            img = Image.open(glob.glob('src/data/examples/*.jpg')[i])
            #transform image
            img = test_transform(img)
            img = torch.unsqueeze(img, 0)
            #predict
            prediction = model(input.to(device))
            prediction = threshold_prediction(prediction)
            plt.subplot(2, 2, i+1)
            plt.title('Prediction: {}'.format(prediction))
            plt.imshow(img)
        plt.show()



#an image name: 0001_000000002.jpg

if __name__ == '__main__':
    main(image_folder='src/data/examples/', single_image=False, image_name=None)

