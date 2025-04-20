# 简单 CNN 模型
import torch
import torch.nn as nn
class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25x25
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6x6
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 输出两个类
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

#mo = FaceCNN()

# from torchinfo import summary
# summary(mo, input_size=(1, 1, 50, 50))  # 注意 batch_size=1
