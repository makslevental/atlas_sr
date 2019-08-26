from torch import optim, nn
from torchvision import transforms
from torchvision.datasets import MNIST

import defaults
from data.databunch import DeviceDataLoader
from models.mnist import MnistCNN
from train.tricks.lr_finder import LRFinder


def test_lr_finder():
    mnist_pwd = "../data"
    batch_size = 256
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(mnist_pwd, train=True, download=True, transform=transform)
    trainloader = DeviceDataLoader(defaults.device, trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = MNIST(mnist_pwd, train=False, download=True, transform=transform)
    testloader = DeviceDataLoader(defaults.device, testset, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    model = MnistCNN().to(defaults.device)
    model = nn.DataParallel(model)

    loss = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    lr_finder = LRFinder(model, optimizer, loss, device="cuda")
    lr_finder.range_test(trainloader, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot()

    lr_finder.range_test(trainloader, val_loader=testloader, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot(skip_end=1)


if __name__ == "__main__":
    test_lr_finder()
