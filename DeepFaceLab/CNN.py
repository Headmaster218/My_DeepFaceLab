import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from DFLIMG import DFLJPG
import numpy as np

# 加载数据并生成标签
def load_data_and_labels(class_paths):
    landmarks = []
    labels = []
    
    for label, class_path in enumerate(class_paths):
        # 遍历 class_path 下的所有图片文件
        file_list = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
        
        for file_name in file_list:
            img_path = os.path.join(class_path, file_name)
            img = DFLJPG.load(img_path)
            
            if img is not None:
                landmarks.append(img.get_landmarks().flatten())  # 提取特征点并展开为 1D
                labels.append(label)  # 每个路径对应的标签
    
    return landmarks, labels

# 数据增强
def augment_landmarks(landmarks, rotation=1, translation=1, noise_std=0.05):
    augmented = []
    for lm in landmarks:
        # 随机旋转
        theta = np.radians(np.random.uniform(-rotation, rotation))
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        lm_rotated = np.dot(lm.reshape(-1, 2), rotation_matrix).flatten()
        
        # 随机平移
        tx = np.random.uniform(-translation, translation)
        ty = np.random.uniform(-translation, translation)
        lm_translated = lm_rotated + np.array([tx, ty] * (len(lm) // 2))
        
        # 添加噪声
        noise = np.random.normal(0, noise_std, lm.shape)
        lm_augmented = lm_translated + noise
        
        augmented.append(lm_augmented)
    return np.array(augmented)

# 定义 CNN 网络结构
class FaceRecognitionCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(FaceRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)  # 卷积层
        self.pool = nn.MaxPool1d(2)  # 池化层
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)  # 第二卷积层
        self.fc1 = nn.Linear(64 * 17, 128)  # 全连接层
        self.fc2 = nn.Linear(128, num_classes)  # 输出层
        

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # 池化层
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # 池化层
        x = x.view(-1, 64 * 17)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 确保在模型的训练函数中，处理的输入数据 batch size 一致
def train_one_epoch(model, train_loader, optimizer, criterion, device):

    # 训练操作
    model.train()
    total_loss = 0
    correct = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == y_batch).sum().item()
    
    accuracy = correct / len(train_loader.dataset)
    return total_loss / len(train_loader), accuracy

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch).sum().item()
    accuracy = correct / len(val_loader.dataset)
    return total_loss / len(val_loader), accuracy

# 定义类别目录
Class0_path = 'D:\\AIFace\\workspace\\data_src\\aligned\\Other'
Class1_path = 'D:\\AIFace\\workspace\\data_src\\aligned\\JM'

# 加载数据
class_paths = [Class0_path, Class1_path]
landmarks, labels = load_data_and_labels(class_paths)

landmarks = augment_landmarks(landmarks=landmarks)

# 数据预处理与划分
X_train, X_val, y_train, y_val = train_test_split(landmarks, labels, test_size=0.4, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# 将数据形状从 (batch_size, 136) 转换为 (batch_size, 2, 68) 以适应 CNN 输入
X_train = X_train.view(-1, 2, 68)  # 2 (channels), 68 (features)
X_val = X_val.view(-1, 2, 68)

# 获取每个类别的样本索引
class_indices = {label: np.where(y_train.numpy() == label)[0] for label in set(y_train.numpy())}

# 确保每个类别的样本数量相同（均衡批次）
min_class_size = min(len(indices) for indices in class_indices.values())
balanced_indices = []

# 对每个类别进行均匀采样，确保每个批次中各类别的样本数相同
for label, indices in class_indices.items():
    sampled_indices = np.random.choice(indices, min_class_size, replace=False)
    balanced_indices.extend(sampled_indices)

# 打乱顺序并创建自定义的训练数据加载器
np.random.shuffle(balanced_indices)

train_loader = DataLoader(TensorDataset(X_train[balanced_indices], y_train[balanced_indices]), batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, drop_last=True)
# 初始化模型
input_channels = 2  # 每个特征点的 (x, y) 坐标
num_classes = 2  # 二分类问题
model = FaceRecognitionCNN(input_channels, num_classes)

# 使用 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# model.load_state_dict(torch.load("face_recognition_cnn_model_final.pth", map_location=device))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

import matplotlib.pyplot as plt
import torch

# 开始训练
# 开始训练
epochs = 1000
train_losses, val_losses = [], []

for epoch in range(epochs):
    # 训练和验证步骤
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 每 50 个周期保存一次模型
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"face_recognition_cnn_model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}.")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

# 最终保存模型
torch.save(model.state_dict(), "face_recognition_cnn_model_final.pth")
print("Final model has been saved.")


