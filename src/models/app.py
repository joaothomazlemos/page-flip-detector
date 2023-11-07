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
import gradio as gr #interface



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
        prediction_prob = model(input.to(device))
        prediction = threshold_prediction(prediction_prob)
        plt.title('Prediction: {}'.format(prediction))
        plt.imshow(img)
        plt.show()
    else:

    #create a grid for the predictions for all the images in the example folder if it is the images in a folder to predict
        for i in range(0, len(glob.glob('src/data/examples/*.jpg'))-1):
            img = Image.open(glob.glob('src/data/examples/*.jpg')[i])
            #transform image
            input = test_transform(img)
            input = torch.unsqueeze(input, 0)
            #predict
            prediction = model(input.to(device))
            prediction = threshold_prediction(prediction)
            plt.subplot(2, 2, i+1)
            plt.title('Prediction: {}'.format(prediction))
            plt.imshow(img)
        plt.show()

def gr_main(img):
    """
    Predicts whether a book page is being flipped or not, based on an input image.

    Args:
        

    Returns:
        None
    """
    #set device avaible
    device = get_device()
    
    model = torch.load('C:\\Users\\joaot\\1 notebooks\\APZIVA\\docReader_project4\\flipping-book-detection\\src\\data\\models\\MobileNetV2\\model.pth')
    #load dictionary of model state weitghts
    model.load_state_dict(torch.load('C:\\Users\\joaot\\1 notebooks\\APZIVA\\docReader_project4\\flipping-book-detection\\src\\data\\models\\MobileNetV2\\checkpoint.pt'))
    #set model to evaluation mode
    model.eval()

    #set model to device
    model.to(device)
    #getting the class of the pred


    #plot the image with predicted label
    #0 = flipping the page
    #1 = not flipping the page

    #img = Image.open(img)
    
    #transform image
    input = test_transform(img)
    input = torch.unsqueeze(input, 0)
    #predict
    prediction_prob = model(input.to(device))
    prediction = threshold_prediction(prediction_prob)
    #get only the value of the probability value
    prediction_prob = prediction_prob.item() if prediction == 'Not-Flipping' else 1 - prediction_prob.item()
    print('Prediction: {}'.format(prediction))
    return '{}'.format(prediction), '{:.2%}'.format(prediction_prob)




demo = gr.Interface(
    fn=gr_main,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability")
    ],
    flagging_options=["correct", "incorrect"],
    examples=[
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0001_000000002.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0002_000000017.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0001_000000009.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0001_000000014.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0001_000000020.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0001_000000024.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0002_000000012.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0002_000000013.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0002_000000015.jpg"),
        os.path.join(os.path.abspath(''), "flipping-book-detection/src/models/examples/0002_000000017.jpg"),
        
    ],
)

#an image name: 0001_000000002.jpg

if __name__ == '__main__':
    demo.launch(show_api=True, share=True)
    #main(image_folder='src/data/examples/', single_image=False, image_name=None)

    #single images
    #main(image_folder='src/data/examples/', single_image=True, image_name='0001_000000002.jpg')
    #main(image_folder='src/data/examples/', single_image=True, image_name='0002_000000017.jpg') e esse3


