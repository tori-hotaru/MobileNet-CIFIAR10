import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from MobileNetV1 import *
from MobileNetV2 import *
from MobileNetV3 import *

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    net = MobileNetV1(3, 10).to(device)
    # net = MobileNetV2(2, 10, 1.0).to(device)
    # net = mobilenetv3(mode='small', n_class=10, input_size=224, dropout=0.2, width_mult=1.0).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 5 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

        epoch += 1
