import torch
import torch.nn
import torch.nn.functional


def flatten(x):
    # read in N, C, H, W
    N = x.shape[0]
    return x.view(N, -1)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return flatten(x)


class ExpConvNet(torch.nn.Module):
    def __init__(self):
        super(ExpConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(torch.nn.functional.relu(x))
        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(torch.nn.functional.relu(x))
        x = flatten(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x


path = 'training.pt'
pretrained = torch.load(path)

model = ExpConvNet()
model.load_state_dict(pretrained)

import collections

new_pretrained = collections.OrderedDict()
for name, value in model.state_dict().items():
    new_pretrained[name] = value.to(torch.int8)

print(new_pretrained)

model.load_state_dict(new_pretrained)
torch.save(model.state_dict(), 'test.pt')
