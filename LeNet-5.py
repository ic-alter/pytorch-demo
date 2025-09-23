# LeNet-5 on MNIST (PyTorch)
# - 经典结构: Conv(6)-AvgPool-Conv(16)-AvgPool-FC(120)-FC(84)-FC(10)
# - 激活函数使用 Tanh，池化使用 AvgPool，输入从 28x28 通过 pad -> 32x32
# - 几个 epoch 通常即可在 MNIST 上达到 98%+ 准确率

import os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# ========== 随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# ========== 配置 ==========
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                      else "cpu")
batch_size = 128
epochs = 8
lr = 0.01
momentum = 0.9
num_workers = 2 if os.name != "nt" else 0  # Windows 下设 0 更稳

# ========== 数据集 ==========
# LeNet-5 期望输入 32x32，这里对 28x28 的 MNIST 在四周 pad 2 像素
transform = transforms.Compose([
    transforms.Pad(2),              # 28x28 -> 32x32
    transforms.ToTensor(),          # [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 统计均值/方差
])

train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# ========== 模型：LeNet-5 ==========
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # C1: 1x32x32 -> 6x28x28 (5x5 conv, no padding)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=True)
        # S2: AvgPool 6x28x28 -> 6x14x14
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C3: 6x14x14 -> 16x10x10
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=True)
        # S4: AvgPool 16x10x10 -> 16x5x5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # C5: 16x5x5 -> 120x1x1  (经典 LeNet-5 的 conv 当作全连接)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=True)

        # F6: 120 -> 84
        self.fc1 = nn.Linear(120, 84)
        # OUTPUT: 84 -> 10
        self.fc2 = nn.Linear(84, num_classes)

        # 经典 LeNet 使用 Tanh
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool1(x)
        x = self.act(self.conv2(x))
        x = self.pool2(x)
        x = self.act(self.conv3(x))        # [B, 120, 1, 1]
        x = x.view(x.size(0), -1)          # [B, 120]
        x = self.act(self.fc1(x))          # [B, 84]
        x = self.fc2(x)                    # [B, 10]
        return x

model = LeNet5().to(device)
print(model)

# 参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.3f}M")

# ========== 优化器 & 损失 ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)

# 可选：学习率衰减（余弦或 step）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ========== 训练/评测函数 ==========
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        pbar.set_description(f"loss {running_loss/total:.4f} | acc {correct/total:.4f}")
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return loss_sum/total, correct/total

# ========== 训练循环 ==========
best_acc = 0.0
save_path = "lenet5_mnist.pt"
t0 = time.time()
for epoch in range(1, epochs+1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    scheduler.step()

    print(f"[Epoch {epoch:02d}] "
          f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
          f"test_loss={val_loss:.4f} test_acc={val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"model": model.state_dict(),
                    "acc": best_acc,
                    "epoch": epoch}, save_path)

t1 = time.time()
print(f"Done. Best test acc: {best_acc*100:.2f}%. Time: {(t1-t0):.1f}s. Model saved to {save_path}")

# ========== 载入与单样本推理 ==========
@torch.no_grad()
def quick_infer_sample():
    model.eval()
    img, label = test_set[0]
    x = img.unsqueeze(0).to(device)
    logits = model(x)
    pred = logits.argmax(dim=1).item()
    print(f"GT label: {label}, Pred: {pred}")

quick_infer_sample()  # 需要时取消注释