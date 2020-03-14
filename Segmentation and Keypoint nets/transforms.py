import torch
from torchvision import transforms, utils
# tranforms

class Normalize(object):
    """Normalizes keypoints.
    """
    def __init__(self, min_keypts_val, max_keypts_val):
        self.min_keypts_val = min_keypts_val
        self.max_keypts_val = max_keypts_val

    def __call__(self, sample):

        image, key_pts = sample['image'], sample['keypoints']

        ##############################################################
        # TODO: Implemnet the Normalize function, where we normalize #
        # the image from [0, 255] to [0,1] and keypoints from [0, 96]#
        # to [-1, 1]                                                 #
        ##############################################################
        image = image / 255.0
        
        # key_pts = 2 * (key_pts - self.min_keypts_val) / \
        #               (self.max_keypts_val - self.min_keypts_val) - 1
        
        key_pts = (key_pts - 48.0) / 48.0
        ##############################################################
        # End of your code                                           #
        ##############################################################
        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        return {'image': torch.from_numpy(image).float(),
                'keypoints': torch.from_numpy(key_pts).float()}
