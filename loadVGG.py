import time
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision
import torch.utils.data.sampler

NUM_TRAIN = 49000

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

cifar10_train = torchvision.datasets.CIFAR10('~/DATA/cifar', train=True, download=True, transform=transform)
loader_train = torch.utils.data.DataLoader(cifar10_train, batch_size=64,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler([range(NUM_TRAIN)]))

cifar10_val = torchvision.datasets.CIFAR10('./cs231n/datasets', train=True, download=True, transform=transform)
loader_val = torch.utils.data.DataLoader(cifar10_val, batch_size=64,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler([range(NUM_TRAIN, 50000)]))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)
