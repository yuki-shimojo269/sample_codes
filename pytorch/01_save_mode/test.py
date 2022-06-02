import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import Net
import os


class Test():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

        testset = torchvision.datasets.MNIST(root='./data',
                                             train=False,
                                             download=True,
                                             transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=100,
                                                      shuffle=False,
                                                      num_workers=2)

        self.net = Net()
        self.net.to(self.device)
        self.get_model_weight()

    def get_model_weight(self):
        if os.path.isfile('Net_encoder.pth') and os.path.isfile('Net_decoder.pth'):
            self.net.Encoder.load_state_dict(torch.load('Net_encoder.pth'))
            self.net.Decoder.load_state_dict(torch.load('Net_decoder.pth'))

    def test_mode(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for (images, labels) in self.testloader:
                images = images.to(self.device)
                outputs = self.net(images)
                outputs = outputs.to('cpu')
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: {:.2f} %%'.format(100 * float(correct / total)))