import torch
import torch.nn as nn
from constants import *


# Custom layer to flatten the output in 1-D vector
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()

        self.base = nn.Sequential(
            nn.Conv2d(nCHANNELS * nFRAMES, 64, kernel_size = (7, 7), stride = (2, 2), padding = (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            nn.Conv2d(64, 64, kernel_size = (5, 5), stride = (1, 1), padding = (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(128, 128, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(128, 256, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(256, 512, kernel_size = (3, 3), stride = (1, 1), padding = (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
            )

        # Pretrained Resnet18 can also be used.
        # from torchvision import models
        # self.base = nn.Sequential(*list(models.resnet18(pretrained = True).children())[:-1]).cuda()

        self.flatter = Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 512),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 512),
            nn.ReLU()
            )
        self.out = nn.Linear(in_features = 512, out_features = nACTIONS)

    def forward(self, x):
        # x = x.view(-1, 3, IMG_HEIGHT, IMG_WIDTH)
        x = self.base(x)
        x = self.flatter(x)
        # x = x.view(-1, nFRAMES * 512)
        x = self.fc(x)
        x = self.out(x)
        return x

    def save(self, filename):
        # Switch network to CPU before saving to avoid issues.
        torch.save(self.cpu().state_dict(), filename)

    def load(self, filename):
        # Load state dictionary from saved file
        self.load_state_dict(torch.load(filename, map_location = 'cpu'))

    def copy(self, freeze = True):
        # Create a copy of self
        copied = QNetwork()
        copied.load_state_dict(self.state_dict())

        # Freeze its parameters
        if freeze:
            for params in copied.parameters():
                params.requires_grad = False

        return copied
