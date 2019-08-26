import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST

import defaults
from data.databunch import DeviceDataLoader
from models.mnist import MnistCNN
from train.tricks.one_cycle_lr import OneCycleLR


def test_one_cycle():
    mnist_pwd = "../data"
    batch_size = 4096
    epochs = 100
    num_steps = 100
    log_interval = 10
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(mnist_pwd, train=True, download=True, transform=transform)
    train_loader = DeviceDataLoader(defaults.device, trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    loss = nn.NLLLoss()
    model = MnistCNN().to(defaults.device)
    model = nn.DataParallel(model)
    print(model.device_ids)

    start_lr = 2.75E-01
    start_mom = 0.5

    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, momentum=start_mom)
    scheduler = OneCycleLR(optimizer, num_steps=num_steps, lr_range=(start_lr, 1.0), momentum_range=(0.1, start_mom))
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            out = model(x)
            loss_val = loss(out, y)
            loss_val.backward()
            optimizer.step()
            scheduler.step()

            if step % log_interval == 0:
                print(step, loss_val, optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["momentum"])



if __name__ == "__main__":
    test_one_cycle()
