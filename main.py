import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import *
import hydra
import numpy as np
import matplotlib.pyplot as plt
import logging


log = logging.getLogger(__name__)


def transpose_and_unnormalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    x = np.transpose(x, axes=(1,2,0))
    x = x * std + mean
    return x


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    batch_size = cfg.train.batch_size

    trainset = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=cfg.data_dir, train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sample_data = next(iter(trainloader))
    inputs, labels = sample_data[0].to(device), sample_data[1].to(device)
    x = inputs.cpu().numpy()
    for i in range(x.shape[0]):
        plt.clf()
        plt.imshow(transpose_and_unnormalize(x[i], cifar10_mean, cifar10_std))
        plt.title(classes[labels[i]])
        plt.savefig(f"{i:03d}.png")

    model = ViT(cfg)
    model.to(device)

    log.info(model)
    criterion = nn.CrossEntropyLoss()
    optimizer =  torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    loss_history = []
    train_acc_history = []
    test_acc_history = []

    for epoch in range(cfg.train.num_epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        model.train()
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        average_loss = running_loss / len(trainloader)
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total

        log.info(f'Epoch {epoch + 1}: Loss: {average_loss:.3f} Train Accuracy: {train_acc:.3f} Test Accuracy: {test_acc:.3f}')

        loss_history.append(average_loss)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    plt.clf()
    plt.plot(loss_history)
    plt.savefig("loss.png")

    plt.clf()
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(test_acc_history, label='Test Acc')
    plt.legend()
    plt.savefig("accuracy.png")

    log.info('Finished Training')
    torch.save(model.state_dict(), "cifar10_vit.pth")


if __name__ == '__main__':
    main()
