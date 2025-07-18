# Brain Tumor MRI Classification with PyTorch

This project implements a brain tumor classifier using a pretrained ResNet-18 model in PyTorch. It uses a publicly available dataset of T1-weighted MRI images categorized into four classes: **glioma tumor**, **meningioma tumor**, **pituitary tumor**, and **no tumor**.

---

## Dataset

We use the **Brain Tumor Classification (MRI)** dataset by Ghaffar et al., available for free (no login required) on [Mendeley Data](https://data.mendeley.com/datasets/w4sw3s9f59/1):

 Contains: `Training/` and `Testing/` folders with subdirectories:
  - `glioma_tumor/`
  - `meningioma_tumor/`
  - `pituitary_tumor/`
  - `no_tumor/`

Download link:  
**https://data.mendeley.com/datasets/w4sw3s9f59/1**

After downloading, extract and place the data in your project directory. You can rename the `Training/` folder to `data/` or specify it with `--data_dir`.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classifier.git
   cd brain-tumor-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

Dependencies include:
- `torch`
- `torchvision`
- `Pillow`

---

## Training

Train a ResNet-18 model using your dataset:

```bash
python train.py   --data_dir ./data   --epochs 10   --batch_size 32   --learning_rate 0.001   --output_model tumor_model.pth
```

- The script splits the data into training and validation (80/20).
- Model accuracy and loss are printed each epoch.
- The best model is saved to `tumor_model.pth`.

---

## Inference

Classify a new image with the trained model:

```bash
python inference.py   --model_path tumor_model.pth   --image_path ./data/glioma_tumor/example1.jpg
```

Output:
```
Predicted class: glioma_tumor
```

---

## Project Structure

```
.
├── data_loader.py        # Custom PyTorch dataset and DataLoader
├── model.py              # TumorClassifier using pretrained ResNet18
├── train.py              # Training script
├── inference.py          # Inference script for new images
├── requirements.txt      # Dependencies
└── README.md             # Project overview and instructions
```

---

## Citation

Dataset:

> Ghaffar, A. (2024), Brain Tumor Classification (MRI), Mendeley Data, V1, https://doi.org/10.17632/w4sw3s9f59.1

