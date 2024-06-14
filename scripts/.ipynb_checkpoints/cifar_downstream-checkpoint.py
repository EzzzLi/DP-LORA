import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def set_seed(seed_value):
    # 设置 PyTorch 随机数种子
    torch.manual_seed(seed_value)
    # 设置 CUDA 随机数种子
    torch.cuda.manual_seed_all(seed_value)
    # 针对Cudnn的优化，设置随机数种子
    torch.backends.cudnn.deterministic = True
    # 针对Cudnn的优化，禁用随机数种子
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, file_path='./cifar_eps10.pt', transform=None):
        data = torch.load(file_path)
        self.image = data['image']
        self.label = data['class_label']
        if transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # sample = self.data[idx]
        image = self.transform(self.image[idx])
        return image, self.label[idx]
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 输入通道数3，输出通道数32，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 输入通道数32，输出通道数64，卷积核大小3x3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 输入通道数64，输出通道数128，卷积核大小3x3
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)  # 2x2窗口的最大池化
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 输入大小为128x4x4，输出大小为512
        self.fc2 = nn.Linear(512, 256)  # 输入大小为512，输出大小为256
        self.fc3 = nn.Linear(256, 10)  # 输入大小为256，输出大小为10（CIFAR-10共有10个类别）
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))  # 第一层卷积 + ReLU + 池化
        x = self.dropout(self.pool(F.relu(self.conv2(x))))  # 第二层卷积 + ReLU + 池化
        x = self.dropout(self.pool(F.relu(self.conv3(x))))  # 第三层卷积 + ReLU + 池化
        x = x.view(-1, 128 * 4 * 4)  # 将特征图展平成一维向量
        x = F.relu(self.fc1(x))  # 第一层全连接 + ReLU
        x = F.relu(self.fc2(x))  # 第二层全连接 + ReLU
        x = self.fc3(x)  # 输出层
        return x






transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),     # 随机水平翻转
    transforms.ToTensor(),                  # 转为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
])


set_seed(0)

# 加载训练集和测试集
train_dataset = CustomDataset(transform=train_transform)
test_dataset = datasets.CIFAR10(root='/root/autodl-tmp/cifar10/', train=False, transform=transform, download=True)


train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 实例化CNN模型
model = Net().to('cuda')
# model = WideResNet(num_classes=10, widen_factor=4).to('cuda')
# model = ResNet(BasicBlock, [1, 1, 1, 1]).to('cuda')

from wideresnet import Wide_ResNet
# model = Wide_ResNet(28, 10, 0.3, 10).to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=5e-5) #CNN 5e-5
optimizer = optim.Adam(model.parameters(), lr=1e-3)

max_acc = 0
# 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    if epoch % 1 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set: {100 * correct / total}%")
        max_acc = max(max_acc, 100 * correct / total)

print(f"Best accuracy on test set: {max_acc}%")

# # 测试模型
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to('cuda'), labels.to('cuda')
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy on test set: {100 * correct / total}%")