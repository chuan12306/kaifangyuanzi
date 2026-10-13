import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 2. 生成数据
X, y = make_friedman1(
    n_samples=1500,
    n_features=10,
    noise=1.0,
    random_state=42
)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  #测试集占总数据的比例
)

# 4. 标准化 （不必要）
scaler = StandardScaler()   #标准化数据
X_train_scaled = scaler.fit_transform(X_train)   #计算均值、标准差并转换每个数据
X_test_scaled = scaler.transform(X_test)     #用测试集的均值和标准差转换测试集中的数据

# 5. 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
# -1：第一维自动计算，第二维为1。PyTorch 中很多操作要求输入和输出的形状匹配。
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 6. 定义线性回归模型（无隐藏层）
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()   #调用父类的构造函数。
        self.linear = nn.Linear(input_dim, 1)  # 创建一个线性层。输入 10 维，输出 1 维

    def forward(self, x):   #前向传播
        return self.linear(x)

model = LinearRegressionModel(input_dim=10)

# 7. 损失函数和优化器
criterion = nn.MSELoss()    #定义损失函数，用来衡量预测值和真实值差多少
optimizer = optim.Adam(model.parameters(), lr=0.01)    #定义优化器，用来根据损失更新模型参数
#lr：学习率 model.parameters() 会返回模型中所有可学习参数

# 8. 训练模型
epochs = 2000  #训练轮数
train_losses = []   #空表格记录每一轮的损失值

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()   #清空所有参数的梯度。
    y_pred = model(X_train_tensor)   #前向传播
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()   #反向传播
    optimizer.step()    #优化器根据刚才算出的梯度和学习率，修改模型参数。
    train_losses.append(loss.item())   #把当前轮的损失值存进列表。

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 9. 测试与评价
model.eval()
with torch.no_grad():    #不计算梯度
    y_pred_test = model(X_test_tensor).numpy().flatten()    #.flatten()：展平成一维
    y_pred_train = model(X_train_tensor).numpy().flatten()

mse_test = mean_squared_error(y_test, y_pred_test)  #均方误差
rmse_test = np.sqrt(mse_test)   #均方根误差
mae_test = mean_absolute_error(y_test, y_pred_test)    #平均绝对误差
r2_test = r2_score(y_test, y_pred_test)    #比直接猜平均值拟合多少

#对训练集，检查过拟合
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print("\n========== 测试集评价指标 ==========")
print(f"MSE  : {mse_test:.4f}")
print(f"RMSE : {rmse_test:.4f}")
print(f"MAE  : {mae_test:.4f}")
print(f"R²   : {r2_test:.4f}")

print("\n========== 训练集评价指标 ==========")
print(f"MSE  : {mse_train:.4f}")
print(f"R²   : {r2_train:.4f}")

# 10. 可视化：真实值 vs 预测值
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k')  #画散点图，alpha：点的透明度；点的边框颜色
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Ideal y = x')
plt.xlabel('True y')
plt.ylabel('Predicted y')
plt.title('Linear Regression: True vs Predicted (Test Set)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('pred_vs_true.png')
plt.show()