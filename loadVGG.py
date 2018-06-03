import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils.data
import torchvision
import torch.utils.data.sampler

NUM_TRAIN = 49000

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 使用sampler.SubsetRandomSampler(slice) 把训练集分为两部分
cifar10_train = torchvision.datasets.CIFAR10('~/DATA/cifar', train=True, download=True, transform=transform)
loader_train = torch.utils.data.DataLoader(cifar10_train, batch_size=64,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler([range(NUM_TRAIN)]))

cifar10_val = torchvision.datasets.CIFAR10('~/DATA/cifar', train=True, download=True, transform=transform)
loader_val = torch.utils.data.DataLoader(cifar10_val, batch_size=64,
                                         sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                             [range(NUM_TRAIN, 50000)]))
# 测试集
cifar10_test = torchvision.datasets.CIFAR10('~/DATA/cifar', train=False, download=True, transform=transform)
loader_test = torch.utils.data.DataLoader(cifar10_test, batch_size=64)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print_every = 100


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
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float64)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model.train()  # put model to training mode
    t_begin = time.time()
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):

            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float64)

            scores = model(x)
            # criterion
            loss = torch.nn.functional.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                t_elapse = time.time() - t_begin
                print('Elapsed %.4f s, Epoch %d,  Iteration %d, loss = %.4f' % (t_elapse, e, t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()


import Model.vgg_modules
import Model.Fixedvgg

PATH = '../pretrain_model/model_best.pth.tar'  # change the path
checkpoint = torch.load(PATH)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_prec1']

VGG19 = Model.vgg_modules.vgg19()
VGG19 = torch.nn.DataParallel(VGG19)
VGG19.load_state_dict(checkpoint['state_dict'])

FixedVGG19 = Model.Fixedvgg.fixed_vgg19()
FixedVGG19 = torch.nn.DataParallel(FixedVGG19)
FixedVGG19.load_state_dict(checkpoint['state_dict'])

print("\nFixed VGG11 Accuracy:")
check_accuracy_part34(loader_test, FixedVGG19)
print("\nFloadt VGG11 Accuracy:")
check_accuracy_part34(loader_test, VGG19)

# fine tune FixedVGG19
learning_rate = 2e-5
optimizer = torch.optim.Adam(params=FixedVGG19.parameters(), lr=learning_rate)
train_part34(FixedVGG19, optimizer, epochs=5)

print("\nFinetune Fixed VGG11 Accuracy:")
check_accuracy_part34(loader_test, FixedVGG19)
print("\nFloadt VGG11 Accuracy:")
check_accuracy_part34(loader_test, VGG19)
