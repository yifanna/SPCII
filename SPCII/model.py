import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from thop import profile, clever_format
import matplotlib.pyplot as plt
import os
import re

from mbv2_SPCII import MBV2_SPCII

batch_size = 64
num_epochs = 200
learning_rate = 0.001


transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" )
model = MBV2_SPCII(num_classes=10, width_mult=1.0).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)



restore_ckpt = "C:/Users/admin/Desktop/SPCII/model_weights_spcii-cifar300.pth"
if os.path.exists(restore_ckpt):
    # Load the model weights from the checkpoint
    model.load_state_dict(torch.load(restore_ckpt))
    print(f"Loaded checkpoint from: {restore_ckpt}")

    # Extract the epoch number from the checkpoint file name
    checkpoint_name = os.path.basename(restore_ckpt)
    epoch_str = ''.join(filter(str.isdigit, checkpoint_name))
    if epoch_str:
        resume_epoch = int(epoch_str)
        start_epoch = resume_epoch + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 1
        print("Epoch information not found in the checkpoint file name. Starting from epoch 1.")
else:
    print(f"Checkpoint not found at: {restore_ckpt}")
    start_epoch = 1

# Continue with the rest of your code, starting from the training loop
for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'model_weights_spcii-cifar10_1.pth')

model.eval()
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

