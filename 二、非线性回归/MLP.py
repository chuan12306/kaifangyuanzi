import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time

# 1. 固定随机种子
np.random.seed(42)
torch.manual_seed(42)

# 2. 生成数据
X, y = make_friedman1(
    n_samples=1500,
    n_features=10,
    noise=1.0,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 3. 定义两个模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32)):
        super().__init__()
        layers = []   #创建一个空列表，用来存放每一层。
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())   #引入非线性，ReLU 是一个函数：ReLU(x)=max(0,x)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 4. 训练函数
def train_model(model, X_train, y_train, epochs=2000, lr=0.01):
    criterion = nn.MSELoss()    #损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)   #优化器
    train_losses = []
    start_time = time.time()    #time.time() 返回当前时间

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()   #清空梯度
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()    #反向传播，算梯度
        optimizer.step()    #更新参数
        train_losses.append(loss.item())      #记录损失

    train_time = time.time() - start_time
    return train_losses, train_time


def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(X).numpy().flatten()
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return y_pred, {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# 5. 训练线性模型
linear_model = LinearRegressionModel(input_dim=10)
linear_losses, linear_time = train_model(linear_model, X_train_tensor, y_train_tensor)

linear_pred_train, linear_train_metrics = evaluate_model(linear_model, X_train_tensor, y_train)
linear_pred_test, linear_test_metrics = evaluate_model(linear_model, X_test_tensor, y_test)

# 6. 训练 MLP 模型
mlp_model = MLPModel(input_dim=10, hidden_dims=(64, 32))
mlp_losses, mlp_time = train_model(mlp_model, X_train_tensor, y_train_tensor)

mlp_pred_train, mlp_train_metrics = evaluate_model(mlp_model, X_train_tensor, y_train)
mlp_pred_test, mlp_test_metrics = evaluate_model(mlp_model, X_test_tensor, y_test)

# 7. 打印结果
def print_metrics(name, metrics):
    print(f"\n===== {name} =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

print_metrics("线性模型 - 训练集", linear_train_metrics)
print_metrics("线性模型 - 测试集", linear_test_metrics)
print_metrics("MLP 模型 - 训练集", mlp_train_metrics)
print_metrics("MLP 模型 - 测试集", mlp_test_metrics)

print(f"\n线性模型训练时间: {linear_time:.2f}s")
print(f"MLP 模型训练时间: {mlp_time:.2f}s")

# 8. 参数量统计
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#返回一个迭代器，遍历模型中所有参数张量。
print(f"\n线性模型参数量: {count_params(linear_model)}")
print(f"MLP 模型参数量: {count_params(mlp_model)}")

# 9. 可视化 1：训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(linear_losses, label='Linear Regression', alpha=0.8)
plt.plot(mlp_losses, label='MLP (64, 32)', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss (Train)')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.savefig('loss_comparison.png')
plt.show()

# 10. 可视化 2：预测值 vs 真实值
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, pred, name, r2 in zip(
    axes,
    [linear_pred_test, mlp_pred_test],
    ['Linear Regression', 'MLP (64, 32)'],
    [linear_test_metrics['R2'], mlp_test_metrics['R2']]
):
    ax.scatter(y_test, pred, alpha=0.6, edgecolors='k')
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            'r--', lw=2, label='Ideal y = x')
    ax.set_xlabel('True y')
    ax.set_ylabel('Predicted y')
    ax.set_title(f'{name} (Test R² = {r2:.4f})')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('pred_vs_true_comparison.png')
plt.show()
