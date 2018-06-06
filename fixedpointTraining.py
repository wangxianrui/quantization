import math
import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data
import torch.utils.data.sampler
import torchvision

import config

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 使用sampler.SubsetRandomSampler(slice) 把训练集分为两部分
cifar10_train = torchvision.datasets.CIFAR10('~/DATA/cifar', train=True, download=True, transform=transform)
loader_train = torch.utils.data.DataLoader(cifar10_train, batch_size=64,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               [x for x in range(config.NUM_TRAIN)]))

cifar10_val = torchvision.datasets.CIFAR10('~/DATA/cifar', train=True, download=True, transform=transform)
loader_val = torch.utils.data.DataLoader(cifar10_val, batch_size=64,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                             [x for x in range(config.NUM_TRAIN, 50000)]))
# 测试集
cifar10_test = torchvision.datasets.CIFAR10('~/DATA/cifar', train=False, download=True, transform=transform)
loader_test = torch.utils.data.DataLoader(cifar10_test, batch_size=64)


def flatten(x):
    # read in N, C, H, W
    N = x.shape[0]
    return x.view(N, -1)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return flatten(x)


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=config.device)
            y = y.to(device=config.device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train_part34(model, optimizer, epochs=1):
    model = model.to(device=config.device)  # move the model parameters to CPU/GPU
    if config.device.type == 'cuda':
        model = torch.nn.DataParallel(model)
    model.train()  # put model to training mode
    t_begin = time.time()
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):

            x = x.to(device=config.device)
            y = y.to(device=config.device)

            scores = model(x)
            # criterion
            loss = torch.nn.functional.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % config.print_every == 0:
                t_elapse = time.time() - t_begin
                print('Elapsed %.4f s, Epoch %d,  Iteration %d, loss = %.4f' % (t_elapse, e, t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()


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

        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.constant_(self.conv1.bias, 0)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.constant_(self.conv2.bias, 0)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)

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


model = ExpConvNet()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
train_part34(model, optimizer, epochs=5)

torch.save(model.state_dict(), 'training.pt')


class Quant:
    def linear(input, bits):
        assert bits >= 1, bits
        if bits == 1:
            return torch.sign(input) - 1
        sf = torch.ceil(torch.log2(torch.max(torch.abs(input))))
        delta = math.pow(2.0, -sf)
        bound = math.pow(2.0, bits - 1)
        min_val = - bound
        max_val = bound - 1
        rounded = torch.floor(input / delta)

        clipped_value = torch.clamp(rounded, min_val, max_val) * delta
        return clipped_value


class quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits, quant_func):
        # Define a constant
        ctx.bits = bits
        # ctx.save_for_backward(x)
        clipped_value = quant_func(x, bits)
        return clipped_value

    def backward(ctx, grad_output):
        grad_x = grad_output.clone()
        return grad_x, None, None


class activation_quantization(torch.nn.Module):
    def __init__(self, bits=8, quant_func=Quant.linear):
        super(activation_quantization, self).__init__()
        self.bits = bits
        self.func = quant_func

    def forward(self, inputActivation):
        return quantization.apply(inputActivation, self.bits, self.func)


class FixedLayerConvNet(torch.nn.Module):
    def __init__(self):
        super(FixedLayerConvNet, self).__init__()
        self.bits = 8
        self.conv1 = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 10)
        self.quant = activation_quantization(8, Quant.linear)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(torch.nn.functional.relu(x))
        x = self.quant(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(torch.nn.functional.relu(x))
        x = flatten(x)
        x = self.quant(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.quant(x)
        x = self.fc2(x)
        return x


fix_model = FixedLayerConvNet()
direct_fix_model = FixedLayerConvNet()
direct_fix_model = direct_fix_model.to(config.device)

PATH = 'training.pt'
fix_model.load_state_dict(torch.load(PATH))
direct_fix_model.load_state_dict(torch.load(PATH))

learning_rate = 2e-6
optimizer = torch.optim.Adam(params=fix_model.parameters(), lr=learning_rate)
train_part34(fix_model, optimizer, epochs=10)

print("Finetune Fixed Point Accuracy:")
check_accuracy_part34(loader_test, fix_model)
print("\nDirect Fixed Point Accuracy:")
check_accuracy_part34(loader_test, direct_fix_model)
print("\nOriginal Floating Point Accuracy:")
check_accuracy_part34(loader_test, model)
