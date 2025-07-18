# visualizer.py
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def visualize_samples(data_dir, class_names=None, num_samples=4):
    """
    Displays sample images from each class in the dataset.
    """
    if class_names is None:
        class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    plt.figure(figsize=(12, 3 * len(class_names)))
    img_count = 1

    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        images = [img for img in os.listdir(cls_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]

        for img_name in images:
            img_path = os.path.join(cls_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            plt.subplot(len(class_names), num_samples, img_count)
            plt.imshow(img)
            plt.title(f"{cls}")
            plt.axis('off')
            img_count += 1

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize sample brain tumor images from each class.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory with class subfolders.')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples per class to display.')
    args = parser.parse_args()

    visualize_samples(args.data_dir, num_samples=args.num_samples)

if __name__ == "__main__":
    main()
