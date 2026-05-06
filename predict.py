"""
单张图片预测
用法：python predict.py --image_path 图片路径
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import torchvision.models as models
import torch.nn as nn

def load_model(model_path, num_classes=2):
    """加载训练好的模型"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(image_path, model, class_names):
    """预测单张图片"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    print(f"图片：{image_path}")
    print(f"预测类别：{class_names[predicted_class]}")
    print(f"置信度：{confidence*100:.2f}%")
    
    for i, name in enumerate(class_names):
        print(f"  {name}: {probabilities[0][i].item()*100:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='图片路径')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='模型路径')
    parser.add_argument('--classes', type=str, nargs='+', default=['狗', '猫'], help='类别名称')
    args = parser.parse_args()
    
    model = load_model(args.model_path, len(args.classes))
    predict_image(args.image_path, model, args.classes)

if __name__ == '__main__':
    main()