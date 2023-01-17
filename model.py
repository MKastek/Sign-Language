import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super().__init__()
        # input 28x28
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),  bias=False)
        self.conv2_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*100*100, num_classes)
        self.fc1_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        flat = conv2.reshape(conv2.shape[0], -1)
        logits = self.fc1_bn(self.fc1(flat))
        return F.log_softmax(logits, dim=1)


if __name__ == '__main__':
    cnn = CNN()
    print(summary(cnn, input_size=(3, 100, 100)))