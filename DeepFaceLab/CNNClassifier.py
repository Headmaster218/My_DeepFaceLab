import os
import torch
import torch.nn as nn
import shutil
from DFLIMG import DFLJPG

# 定义分类文件夹
input_path = "D:\\AIFace\\workspace\\data_src\\aligned\\aligned03"  # 输入图片目录
output_class0_path = "D:\\AIFace\\workspace\\data_src\\aligned\\Classified_Other"   # 分类到类别 0 的目录
output_class1_path = "D:\\AIFace\\workspace\\data_src\\aligned\\Classified_JM"   # 分类到类别 1 的目录

# 创建输出目录
os.makedirs(output_class0_path, exist_ok=True)
os.makedirs(output_class1_path, exist_ok=True)

# 定义CNN网络结构
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

# 模型参数
input_channels = 2  # 每个特征点的 (x, y) 坐标
num_classes = 2  # 两个类别
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并加载权重
model = FaceRecognitionCNN(input_channels=input_channels, num_classes=num_classes)
model.load_state_dict(torch.load("face_recognition_cnn_model_bbest.pth", map_location=device))
model = model.to(device)
model.eval()

# 处理函数：加载图片并分类
def classify_image(img_path, model, device):
    img = DFLJPG.load(img_path)  # 加载图片
    if img is None or img.get_landmarks() is None:
        return None  # 如果图片无效或没有特征点，返回 None
    
    landmarks = img.get_landmarks().flatten()  # 提取特征点并展平
    landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).to(device).unsqueeze(0)  # 转为张量
    landmarks_tensor = landmarks_tensor.view(landmarks_tensor.size(0), 2, 68)  # 转换为 (batch_size, 2, 68)
    
    with torch.no_grad():
        outputs = model(landmarks_tensor)
        predicted_class = outputs.argmax(1).item()  # 获取预测类别
    return predicted_class

# 遍历输入目录并分类
file_path_list = [
    os.path.abspath(os.path.join(root, f))  # 获取每个文件的绝对路径
    for root, _, files in os.walk(input_path)
    for f in files if f.lower().endswith(('.jpg', '.png'))
]

def move_image_with_unique_name(img_path, output_dir):
    file_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, file_name)
    
    # 如果文件已经存在，添加一个后缀来避免覆盖
    base_name, ext = os.path.splitext(file_name)
    counter = 1
    
    # 检查文件是否存在，并添加后缀
    while os.path.exists(output_path):
        new_file_name = f"{base_name}_{counter}{ext}"
        output_path = os.path.join(output_dir, new_file_name)  # 更新 output_path
        counter += 1
    
    # 移动文件
    shutil.move(img_path, output_path)

# 处理图片
from tqdm import tqdm

for img_path in tqdm(file_path_list, desc="Processing images", unit="image"):
    predicted_class = classify_image(img_path, model, device)
    
    if predicted_class is None:
        print(f"无法处理图片: {img_path}")
        continue
    try:
        # 将图片移动到对应的类别目录，并确保文件名不重复
        if predicted_class == 0:
            move_image_with_unique_name(img_path, output_class0_path)
        elif predicted_class == 1:
            move_image_with_unique_name(img_path, output_class1_path)
    except:
        print(img_path)
