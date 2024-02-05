import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from mbv2_SPCII import MBV2_SPCII
from thop import profile, clever_format
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 为验证集定义转换
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载CIFAR-10验证数据集
val_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 创建模型实例
model = MBV2_SPCII(num_classes=10, width_mult=1.0).to(device)

# 指定已训练模型权重的路径
ckpt_path = "model_weights_spcii-cifar200.pth"  # 替换为你的.pth文件路径
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    print(f"Loaded model weights from: {ckpt_path}")
else:
    print(f"Model weights not found at: {ckpt_path}")
    exit()

# 将模型设置为评估模式
model.eval()

# 在验证集上评估模型
correct = 0
total = 0

with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

accuracy = 100.0 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
input_tensor = torch.randn(1, 3, 224, 224).to(device)  # 根据模型输入尺寸进行调整
flops, params = profile(model, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.3f")
print(f"Number of parameters: {params}")
print(f"FLOPs: {flops}")

cifar10_error = 100.0 - accuracy
print(f"CIFAR-10 Error: {cifar10_error:.2f}%")
