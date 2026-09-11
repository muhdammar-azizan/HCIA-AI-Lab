import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Data preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Download and load the CIFAR-10 dataset.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# Display the images.
dataiter = iter(trainloader)
images, labels = next(dataiter)
# Create a grid to display images.
imshow = torchvision.utils.make_grid(images)

# Convert to a NumPy array and remove normalization.
imshow = imshow.numpy().transpose((1, 2, 0))

# Show images.
plt.imshow(imshow)
plt.show()

# Define the LeNet network structure.
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__() # Call the constructor of the parent class nn.Module.
        # For the first convolutional layer, the number of input channels is 3, the number of output channels is 6, and the size of the convolution kernel is 5 x 5.
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        # For the second convolutional layer, the number of input channels is 6, the number of output channels is 16, and the size of the convolution kernel is 5 x 5.
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # For the first fully-connected layer, the number of input features is 16 x 5 x 5, and the number of output features is 120.
        self.fc1 = nn.Linear(16*5*5, 120)
        # For the second fully-connected layer, the number of input features is 120, and the number of output features is 84.
        self.fc2 = nn.Linear(120, 84)
        # For the third fully-connected layer (output layer), the number of input features is 84, and the number of output features is 10.
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        # Perform 2 x 2 max pooling on x.
        x = torch.max_pool2d(x, 2)
        # x passes through the second convolutional layer, and then applies the tanh activation function.
        x = torch.tanh(self.conv2(x))
        # Perform 2 x 2 max pooling on x.
        x = torch.max_pool2d(x, 2)
        # Flatten x into a one-dimensional vector and input it to the fully-connected layer.
        x = x.view(-1, 16*5*5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        # x passes through the third fully-connected layer (output layer).
        x = self.fc3(x)
        return x

# Instantiate the LeNet network.
net = LeNet()
print(net.parameters)  # View the model parameters.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Check whether there are trained parameters to load.
if os.path.isfile("cifar10.pth"):
    net.load_state_dict(torch.load("cifar10.pth"))

# Store the loss value.
loss_history = []

# Train the network.
# Note: Guide originally uses 50 epochs; reduced to 5 for practical runtime on CPU.
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 200:.3f}')
            running_loss = 0.0

        # Record the loss value.
        loss_history.append(loss.item())

print('Finished Training')

# Draw the loss curve.
plt.plot(loss_history)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss over Training Batches')
plt.show()

# Test the performance of the network with the test set.
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')


