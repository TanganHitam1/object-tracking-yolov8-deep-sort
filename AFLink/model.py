"""
@Author: Du Yunhao
@Filename: model.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 14:13
@Discription: model
"""
import torch
from torch import nn

class TemporalBlock(nn.Module): # TemporalBlock class inherits from nn.Module class (PyTorch) -> TemporalBlock is a neural network module
    def __init__(self, cin, cout): # Constructor for TemporalBlock class
        super(TemporalBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, (7, 1), bias=False) # 2D convolutional layer (input channels, output channels, kernel size, bias)
        self.relu = nn.ReLU(inplace=True) # ReLU activation function
        self.bnf = nn.BatchNorm1d(cout) # Batch normalization layer (1D)
        self.bnx = nn.BatchNorm1d(cout) # Batch normalization layer (1D)
        self.bny = nn.BatchNorm1d(cout) # Batch normalization layer (1D)

    def bn(self, x): # Batch normalization function
        x[:, :, :, 0] = self.bnf(x[:, :, :, 0]) # Batch normalization (1D) for the first channel
        x[:, :, :, 1] = self.bnx(x[:, :, :, 1]) # Batch normalization (1D) for the second channel
        x[:, :, :, 2] = self.bny(x[:, :, :, 2]) # Batch normalization (1D) for the third channel
        return x

    def forward(self, x): # Forward pass function for TemporalBlock class (x is the input tensor)
        x = self.conv(x) # Convolutional layer
        x = self.bn(x) # Batch normalization
        x = self.relu(x) # ReLU activation
        return x


class FusionBlock(nn.Module): # FusionBlock class inherits from nn.Module class (PyTorch) -> FusionBlock is a neural network module
    def __init__(self, cin, cout): # Constructor for FusionBlock class
        super(FusionBlock, self).__init__() 
        self.conv = nn.Conv2d(cin, cout, (1, 3), bias=False) # 2D convolutional layer (input channels, output channels, kernel size, bias)
        self.bn = nn.BatchNorm2d(cout) # Batch normalization layer (2D)
        self.relu = nn.ReLU(inplace=True) # ReLU activation function

    def forward(self, x):
        x = self.conv(x) # Convolutional layer
        x = self.bn(x) # Batch normalization
        x = self.relu(x) # ReLU activation
        return x


class Classifier(nn.Module):
    def __init__(self, cin):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(cin*2, cin//2) # Fully connected layer (input size, output size) -> (input size*2, output size)
        self.relu = nn.ReLU(inplace=True) # ReLU activation function (inplace)
        self.fc2 = nn.Linear(cin//2, 2) # Fully connected layer (input size, output size) -> (input size//2, output size)

    def forward(self, x1, x2): # Forward pass function for Classifier class (x1 and x2 are the input tensors)
        x = torch.cat((x1, x2), dim=1) # Concatenate x1 and x2 along the second dimension
        x = self.fc1(x) # Fully connected layer
        x = self.relu(x) # ReLU activation
        x = self.fc2(x) # Fully connected layer
        return x


class PostLinker(nn.Module): # PostLinker class inherits from nn.Module class (PyTorch) -> PostLinker is a neural network module
    def __init__(self):
        super(PostLinker, self).__init__()
        self.TemporalModule_1 = nn.Sequential( # Sequential container for TemporalBlock layers
            TemporalBlock(1, 32), # TemporalBlock (input channels, output channels)
            TemporalBlock(32, 64), # TemporalBlock (input channels, output channels)
            TemporalBlock(64, 128), # TemporalBlock (input channels, output channels)
            TemporalBlock(128, 256) # TemporalBlock (input channels, output channels)
        )
        self.TemporalModule_2 = nn.Sequential( # Sequential container for TemporalBlock layers
            TemporalBlock(1, 32), # TemporalBlock (input channels, output channels)
            TemporalBlock(32, 64), # TemporalBlock (input channels, output channels)
            TemporalBlock(64, 128), # TemporalBlock (input channels, output channels)
            TemporalBlock(128, 256) # TemporalBlock (input channels, output channels)
        )
        self.FusionBlock_1 = FusionBlock(256, 256) # FusionBlock (input channels, output channels)
        self.FusionBlock_2 = FusionBlock(256, 256) # FusionBlock (input channels, output channels)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1)) # Adaptive average pooling layer (output size)
        self.classifier = Classifier(256) # Classifier (input size)

    def forward(self, x1, x2): # Forward pass function for PostLinker class (x1 and x2 are the input tensors)
        x1 = x1[:, :, :, :3] # x1 is the input tensor (all rows, all columns, all frames, first three channels)
        x2 = x2[:, :, :, :3] # x2 is the input tensor (all rows, all columns, all frames, first three channels)
        x1 = self.TemporalModule_1(x1)  # [B,1,30,3] -> [B,256,6,3]
        x2 = self.TemporalModule_2(x2)  # [B,1,30,3] -> [B,256,6,3]
        x1 = self.FusionBlock_1(x1) # [B,256,6,3] -> [B,256,6,1]
        x2 = self.FusionBlock_2(x2) # [B,256,6,3] -> [B,256,6,1]
        x1 = self.pooling(x1).squeeze(-1).squeeze(-1) # Adaptive average pooling (x1) -> [B,256,1,1] -> [B,256]
        x2 = self.pooling(x2).squeeze(-1).squeeze(-1) # Adaptive average pooling (x2) -> [B,256,1,1] -> [B,256]
        y = self.classifier(x1, x2) # Classifier (x1, x2) -> y (output tensor) [B,2]
        if not self.training: # If the model is not in training mode (evaluation mode)
            y = torch.softmax(y, dim=1) # Softmax activation function (y) -> y (output tensor) [B,2]
        return y


if __name__ == '__main__':
    x1 = torch.ones((2, 1, 30, 3))
    x2 = torch.ones((2, 1, 30, 3))
    m = PostLinker()
    y = m(x1, x2)
    print(y)