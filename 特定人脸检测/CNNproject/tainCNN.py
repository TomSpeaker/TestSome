import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from model import FaceCNN


def train_model(
    dataset_path='dataset',
    model_path='face_cnn_model.pth',
    load_model=False,
    batch_size=32,
    num_epochs=10,
    learning_rate=0.001,
    image_size=50
):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"检测到的类别：{dataset.classes}")

    # 模型、损失、优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceCNN().to(device)
    if load_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ 成功加载模型参数：{model_path}")
    else:
        print("🚀 从头开始训练新模型...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"✅ 模型已保存至：{model_path}")


# 示例调用：
if __name__ == '__main__':
    train_model(load_model=False,num_epochs=15)  # 如果要加载已有模型接着训练，改为 load_model=True
