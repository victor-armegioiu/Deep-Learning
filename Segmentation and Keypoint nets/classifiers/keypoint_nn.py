import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(512, 30)

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return self.model(x)

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
