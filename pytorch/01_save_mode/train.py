import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from model import Net


class Train():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data',
                                              train=True,
                                              download=True,
                                              transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset,
                                                       batch_size=100,
                                                       shuffle=True,
                                                       num_workers=2)

        testset = torchvision.datasets.MNIST(root='./data',
                                             train=False,
                                             download=True,
                                             transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset,
                                                      batch_size=100,
                                                      shuffle=False,
                                                      num_workers=2)

        classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

        self.net = Net()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

    def train_mode(self):
        self.net.to(self.device)

        epochs = 1

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.trainloader, 0):
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)
                outputs = outputs.to('cpu')
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[{:d}, {:5d}] loss: {:.3f}'
                          .format(epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
        torch.save(self.net.Encoder.state_dict(), 'Net_encoder.pth')
        torch.save(self.net.Decoder.state_dict(), 'Net_decoder.pth')
        print('Finished Training')

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