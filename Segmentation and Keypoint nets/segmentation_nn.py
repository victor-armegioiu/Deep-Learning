"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        self.features = models.vgg19(pretrained=True).features
        self.conv = torch.nn.Conv2d(512, num_classes, (1, 1))
        self.upsample = torch.nn.Upsample(240)

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        x = self.features(x)
        x = self.conv(x)
        x = self.upsample(x)

        return x 

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

