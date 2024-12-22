
from torchvision import transforms, datasets

# Create a custom dataset wrapper to include file paths
class ImageNetWithPaths(datasets.ImageNet):
    def __getitem__(self, index):
        # Get the original image and label
        original_tuple = super().__getitem__(index)
        # Get the image path
        path = self.samples[index][0]
        # Return image, label, and path
        return original_tuple + (path,)

