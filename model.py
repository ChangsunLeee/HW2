import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, img):
        x = F.tanh(self.conv1(img))
        x = F.avg_pool2d(x, kernel_size=2)
        x = F.tanh(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2)
        x = torch.flatten(x, 1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self, num_classes=10):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(784, 76)
        self.fc2 = nn.Linear(76, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, img):
        batch_size = img.size(0)
        x = torch.flatten(img, 1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x.view(batch_size, -1)
