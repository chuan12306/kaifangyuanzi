import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1、超参数
CFG = dict(root="./data", num_classes=43, size=32, epochs=10,
           bs=128, lr=1e-3)
mean, std = [0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629]  #开源数据
#   归一化数据的均值和标准差
#2、数据
train_tf = transforms.Compose([
    transforms.Resize((CFG["size"], CFG["size"])),  #缩放图像尺寸
    transforms.RandomRotation(10),   #随机旋转 ±10°
    transforms.ColorJitter(0.2, 0.2),   #改变颜色的亮度和对比度
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([
    transforms.Resize((CFG["size"], CFG["size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

def get_loaders(bs):     #数据加载
    tr = torchvision.datasets.GTSRB(CFG["root"], split="train",
                                    download=True, transform=train_tf)
    te = torchvision.datasets.GTSRB(CFG["root"], split="test",
                                    download=True, transform=test_tf)
    return (DataLoader(tr, bs, shuffle=True, num_workers=0),
            DataLoader(te, bs, shuffle=False, num_workers=0))
#shuffle：训练打乱顺序，测试不打乱

#3、模型
class BaselineCNN(nn.Module):
    def __init__(self, n=43):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),  #全连接层（输入维度，输出维度）
            nn.Linear(256, n),
        )
        #（通道，卷积核个数，卷积核大小）nn.MaxPool2d(2)：2×2 最大池化
    def forward(self, x):
        return self.net(x)

class ImprovedCNN(nn.Module):
    def __init__(self, n=43):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25),
            )
        #nn.BatchNorm2d(o)：批归一化，对每个通道做归一化
        #nn.Dropout(0.25)：训练时随机丢弃 25% 的神经元/特征
        self.net = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, n),
        )
    def forward(self, x):
        return self.net(x)

#4、训练
def run_epoch(model, loader, crit, opt=None):
    #crit：损失函数。opt=None：优化器。默认 None 表示只做评估，不训练。
    train = opt is not None
    model.train() if train else model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)  #把数据搬到 device
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad()    #清空梯度
                loss.backward()
                opt.step()  #更新梯度
            loss_sum += loss.item() * x.size(0)   #加权损失
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

def train(model, trl, tel, epochs, lr):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    hist = {k: [] for k in ["tl", "ta", "vl", "va"]}
    for e in range(epochs):
        tl, ta = run_epoch(model, trl, crit, opt)
        vl, va = run_epoch(model, tel, crit)
        hist["tl"].append(tl); hist["ta"].append(ta)
        hist["vl"].append(vl); hist["va"].append(va)
        print(f"Epoch {e+1}/{epochs} | train loss {tl:.4f} acc {ta:.4f} "
              f"| val loss {vl:.4f} acc {va:.4f}")
    return hist

#5、可视化
def plot(hist, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(hist["tl"], label="train")
    ax[0].plot(hist["vl"], label="val")
    ax[0].set_title(f"{title} Loss")
    ax[0].legend()
    ax[1].plot(hist["ta"], label="train")
    ax[1].plot(hist["va"], label="val")
    ax[1].set_title(f"{title} Acc")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

#6、主流程
if __name__ == "__main__":
    trl, tel = get_loaders(CFG["bs"])

    print("=" * 50)
    print("Training Baseline CNN")
    print("=" * 50)
    base = BaselineCNN(CFG["num_classes"]).to(device)
    hist_base = train(base, trl, tel, CFG["epochs"], CFG["lr"])
    plot(hist_base, "Baseline")

    print("=" * 50)
    print("Training Improved CNN")
    print("=" * 50)
    imp = ImprovedCNN(CFG["num_classes"]).to(device)
    hist_imp = train(imp, trl, tel, CFG["epochs"], 5e-4)
    plot(hist_imp, "Improved")