import torch
import torch.nn as nn

#convolutional neural network
#image shape = 224x224x3
#classes = 2

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), # 3 input channels, 32 kernerls, 3x3 kernel, stride of 1. Stride is the step size of the kernel
            nn.BatchNorm2d(32), # BatchNorm2d normalizes the output of the previous layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2x2 kernel, stride of 2. MaxPool2d is a downsampling operation
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1), # Dropout layer to reduce overfitting

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1), # Dropout layer to reduce overfitting

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1), # Dropout layer to reduce overfitting




            nn.Flatten(), # Flatten the output of the previous layer to a vector
            #nn.linear input size: (Input Width - Kernel size + 2*Padding / Stride) + 1
            #input size = 224x224x3
            #after first conv2d layer = (224 - 3 + 2*1 / 1) + 1 => (224x224)x32
            #after first maxpool2d layer = 112x112x32
            #after second conv2d layer = 112x112x64
            #after second maxpool2d layer = 56x56x64
            #after third conv2d layer = 28x28x64
            #after third maxpool2d layer = 28x28x64
            #after fourth conv2d layer = 14x14x32
            #after fourth maxpool2d layer = 14x14x32
            #after fifth conv2d layer = 7x7x32
            #after fifth maxpool2d layer = 7x7x32


            
            nn.Linear(7*7*32, 1024), # input features, 1024 output features
            nn.ReLU(),
            nn.Dropout(0.1), # Dropout layer to reduce overfitting
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.1), # Dropout layer to reduce overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1), # Dropout layer to reduce overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1), # Dropout layer to reduce overfitting
            nn.Linear(32,1) # 32 input features, 1 output features (0 or 1)


        
            
            
  
        )
        
    def forward(self, x):
        return torch.sigmoid(self.model(x)) # Sigmoid activation function for binary classification BCE LOSS
        



