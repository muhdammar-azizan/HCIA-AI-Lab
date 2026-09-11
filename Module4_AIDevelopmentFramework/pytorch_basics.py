import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
x = torch.rand(4, 3)
print(x)

import numpy as np
print(torch.Tensor(np.array([[1, 2], [3, 4]]))) # Use an array to create a tensor.
print(torch.Tensor([True,False])) # Use bool data to create a tensor.
print(torch.Tensor((1,2,3,4,5))) # Use a tuple to create a tensor.
print(torch.Tensor([i for i in range(10)])) # Use a list to create a tensor.

print(torch.FloatTensor(2,3).type()) # Build a 2 x 3 tensor of the float type.
print(torch.DoubleTensor(2,3).type()) # Build a 2 x 3 tensor of the double type.
print(torch.HalfTensor(2,3).type()) # Build a 2 x 3 tensor of the HalfTensor type.
print(torch.ByteTensor(2,3).type()) # Build a 2 x 3 tensor of the byte type.
print(torch.CharTensor(2,3).type()) # Build a 2 x 3 tensor of the char type.
print(torch.ShortTensor(2,3).type()) # Build a 2 x 3 tensor of the short type.
print(torch.IntTensor(2,3).type()) # Build a 2 x 3 tensor of the Int type.
print(torch.LongTensor(2,3).type()) # Build a 2 x 3 tensor of the long type.

x = torch.zeros(4, 3, dtype=torch.long)
print(x) # All-zero tensor
x = torch.ones(4, 3, dtype=torch.long)
print(x) # All-1 tensor
x = x.new_ones(4, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

x = torch.Tensor(np.array([[1, 2], [3, 4]]))
print(x.shape) # Obtain the shape.
print(x.size())
print(x.dim()) # Number of dimensions
print(x.device) # Hardware
print(x.dtype) # Data type

x = torch.Tensor(np.array([[1, 2], [3, 4]]))

print(x.reshape(4,1)) # reshape
x = x.unsqueeze(2) # Add a dimension.
print(x.shape)
x = x.squeeze(2) # Compress dimensions.
print(x.shape)

print(x.flatten()) # Flatten to one dimension.

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Read images from the MNIST dataset.
train_dataset = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transforms.ToTensor())

# View the image and set the image size.
plt.figure(figsize=(8,8))
i = 1

# Print three subplots of the training set.
for dic in train_dataset.data[:3]:
    plt.subplot(3,3,i)
    plt.imshow(dic)
    plt.axis('off')
    i += 1
plt.show()

import torchvision
import random

train = torchvision.datasets.ImageFolder("./flower_photos_train/") # Load the data.

# Check the dataset attributes.
print("Dataset types")
print(train.classes)
print(train.class_to_idx)
print("Dataset samples")
print(train.samples[0])
print("Image data format")
print(train.extensions[0])

for i in range(5):
    plt.figure()
    random.seed = 2
    num = random.randint(0,3000)
    plt.imshow(train[num][0])
    plt.axis('off')
plt.show()

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Pseudocode - template for building a custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels = self._load_labels(label_file)
        self.image_filenames = sorted(os.listdir(image_dir))
        self.transform = transform

    def _load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = [int(line.strip()) for line in f]
        return labels

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label