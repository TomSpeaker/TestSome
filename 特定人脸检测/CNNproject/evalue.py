import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import FaceCNN  # 你之前定义的 CNN 模型

def predict_image(image_path, model_path='face_cnn_model.pth', image_size=50, threshold=0.9):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    # 加载图像
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # 加上 batch 维度 [1, 1, 50, 50]

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 推理
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probs = torch.softmax(output, dim=1)
        face_prob = probs[0][0].item()  # 假设索引0是“你的脸”

    # 判断是否是“你的脸”
    result = 1 if face_prob >= threshold else 0
    print(f"图片 {image_path} 的预测概率为 {face_prob:.4f}，判断为：{'你的脸' if result == 1 else '不是你的脸'}")
    return result


# 示例调用
if __name__ == '__main__':
    test_image = r'C:\Users\26423\Desktop\pythonTest\AiTest\人体检测\CNNproject\dataset\nonface\face_214.jpg'  # 替换为你的测试图片路径
    print(predict_image(test_image,threshold=0.95))
