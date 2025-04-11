import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 检查CUDA是否可用,没有就用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
# 定义训练数据的转换
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化
])

# 定义测试数据的转换
transform_test = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 标准化
])

# 加载CIFAR-10数据集
def load_data():
    # 加载训练数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # 创建训练数据加载器
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    # 加载测试数据集
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # 创建测试数据加载器
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    return trainloader, testloader

# 定义ResNet18模型
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 定义基本残差块
class BasicBlock(nn.Module):
    expansion = 1  # 用于标识残差块输出通道数的扩展倍数

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        # 如果输入和输出的维度不匹配，则使用1x1卷积调整维度
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 前向传播
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = F.relu(out)
        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64
        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 创建四个残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 全连接层
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 创建一个残差层
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个block使用指定的stride，其余block使用stride=1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 训练函数
def train(model, trainloader, criterion, optimizer, epoch, scheduler):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到GPU

        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        _, predicted = outputs.max(1)  # 获取预测结果
        total += targets.size(0)  # 累计总样本数
        correct += predicted.eq(targets).sum().item()  # 累计正确样本数
        running_loss += loss.item()  # 累计损失

    # 计算训练准确率
    train_acc = 100. * correct / total
    print(f"Epoch {epoch} | Loss: {running_loss / len(trainloader):.4f} | Acc: {train_acc:.2f}%")
    scheduler.step()  # 更新学习率
    return train_acc

# 测试函数
def test(model, testloader, criterion):
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移动到GPU
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失

            _, predicted = outputs.max(1)  # 获取预测结果
            total += targets.size(0)  # 累计总样本数
            correct += predicted.eq(targets).sum().item()  # 累计正确样本数
            test_loss += loss.item()  # 累计损失

    # 计算测试准确率
    test_acc = 100. * correct / total
    print(f"Test Loss: {test_loss / len(testloader):.4f} | Test Acc: {test_acc:.2f}%\n")
    return test_acc

if __name__ == '__main__':
    # 加载数据
    trainloader, testloader = load_data()

    # 初始化模型
    model = ResNet18().to(device)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)  # 使用SGD优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # 使用余弦退火学习率调度器

    # 训练循环
    target_train_acc = 85.0  # 目标训练准确率
    target_test_acc = 85.0  # 目标测试准确率
    for epoch in range(20): # 训练20轮
        train_acc = train(model, trainloader, criterion, optimizer, epoch, scheduler)
        test_acc = test(model, testloader, criterion)




# 运行结果
# D:\Anaconda3\envs\pytorch_gpu\python.exe "D:\Pycharm Projects\深度学习\卷积神经网络CNN\ResNet残差神经网络\ResNet实现CIFAR10图像识别.py"
# Using device: cuda
# Using device: cuda
# Using device: cuda
# Epoch 0 | Loss: 1.9983 | Acc: 30.18%
# Using device: cuda
# Using device: cuda
# Test Loss: 1.6319 | Test Acc: 38.65%
#
# Using device: cuda
# Using device: cuda
# Epoch 1 | Loss: 1.5184 | Acc: 43.95%
# Using device: cuda
# Using device: cuda
# Test Loss: 1.3847 | Test Acc: 49.17%
#
# Using device: cuda
# Using device: cuda
# Epoch 2 | Loss: 1.3273 | Acc: 51.59%
# Using device: cuda
# Using device: cuda
# Test Loss: 1.3065 | Test Acc: 53.82%
#
# Using device: cuda
# Using device: cuda
# Epoch 3 | Loss: 1.1434 | Acc: 59.03%
# Using device: cuda
# Using device: cuda
# Test Loss: 1.1653 | Test Acc: 59.42%
#
# Using device: cuda
# Using device: cuda
# Epoch 4 | Loss: 0.9936 | Acc: 64.60%
# Using device: cuda
# Using device: cuda
# Test Loss: 1.0517 | Test Acc: 63.63%
#
# Using device: cuda
# Using device: cuda
# Epoch 5 | Loss: 0.8792 | Acc: 68.98%
# Using device: cuda
# Using device: cuda
# Test Loss: 1.0295 | Test Acc: 65.86%
#
# Using device: cuda
# Using device: cuda
# Epoch 6 | Loss: 0.7580 | Acc: 73.34%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.9164 | Test Acc: 68.86%
#
# Using device: cuda
# Using device: cuda
# Epoch 7 | Loss: 0.6694 | Acc: 76.73%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.7372 | Test Acc: 75.28%
#
# Using device: cuda
# Using device: cuda
# Epoch 8 | Loss: 0.6052 | Acc: 79.02%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.7523 | Test Acc: 74.06%
#
# Using device: cuda
# Using device: cuda
# Epoch 9 | Loss: 0.5719 | Acc: 80.22%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6830 | Test Acc: 76.85%
#
# Using device: cuda
# Using device: cuda
# Epoch 10 | Loss: 0.5364 | Acc: 81.63%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6289 | Test Acc: 78.60%
#
# Using device: cuda
# Using device: cuda
# Epoch 11 | Loss: 0.5151 | Acc: 82.34%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6042 | Test Acc: 78.78%
#
# Using device: cuda
# Using device: cuda
# Epoch 12 | Loss: 0.4887 | Acc: 83.20%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6221 | Test Acc: 79.07%
#
# Using device: cuda
# Using device: cuda
# Epoch 13 | Loss: 0.4739 | Acc: 83.86%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6465 | Test Acc: 77.84%
#
# Using device: cuda
# Using device: cuda
# Epoch 14 | Loss: 0.4621 | Acc: 84.15%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.7126 | Test Acc: 76.65%
#
# Using device: cuda
# Using device: cuda
# Epoch 15 | Loss: 0.4472 | Acc: 84.68%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6383 | Test Acc: 79.11%
#
# Using device: cuda
# Using device: cuda
# Epoch 16 | Loss: 0.4361 | Acc: 85.17%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.5682 | Test Acc: 81.16%
#
# Using device: cuda
# Using device: cuda
# Epoch 17 | Loss: 0.4246 | Acc: 85.38%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6908 | Test Acc: 77.76%
#
# Using device: cuda
# Using device: cuda
# Epoch 18 | Loss: 0.4173 | Acc: 85.82%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.5711 | Test Acc: 80.96%
#
# Using device: cuda
# Using device: cuda
# Epoch 19 | Loss: 0.4094 | Acc: 85.85%
# Using device: cuda
# Using device: cuda
# Test Loss: 0.6476 | Test Acc: 78.77%
#
#
# 进程已结束，退出代码为 0
