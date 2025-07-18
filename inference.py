# inference.py
import argparse
import torch
from model import TumorClassifier
from torchvision import transforms
from PIL import Image

def predict(model, device, image_path, transform, class_names):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # add batch dimension
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]
    return predicted_class

def main():
    parser = argparse.ArgumentParser(description='Classify a brain tumor MRI image.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model file (state_dict).')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image file.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model and weights
    model = TumorClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Class names in sorted order of folder names
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

    predicted = predict(model, device, args.image_path, transform, class_names)
    print(f'Predicted class: {predicted}')

if __name__ == '__main__':
    main()
