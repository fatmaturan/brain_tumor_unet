# Brain Tumor Segmentation with U-Net

A deep learning project for brain tumor segmentation from MRI images using the U-Net architecture.

## Project Overview

This project implements a U-Net based semantic segmentation model to identify brain tumor regions in MRI images.

The workflow includes:

- Image preprocessing
- Data augmentation
- U-Net model implementation
- Model training
- Prediction
- Evaluation using segmentation metrics

---

## Dataset

Brain MRI segmentation dataset.

Images and masks are paired for supervised semantic segmentation.

---

## Model

- U-Net Architecture
- Encoder-Decoder Structure
- Skip Connections
- Binary Segmentation

Loss Functions:

- Binary Cross Entropy
- Dice Loss

Optimizer:

- Adam

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

---

## Folder Structure

```
brain_tumor_unet
│
├── dataset/
├── models/
├── predictions/
├── notebooks/
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Results

The trained U-Net model successfully segments brain tumor regions from MRI images.

Evaluation metrics include:

- Dice Score
- IoU
- Accuracy
- Loss Curves

---

## Future Improvements

- Attention U-Net
- UNet++
- 3D U-Net
- Better Data Augmentation
- Hyperparameter Optimization

---

## Author

Fatma Turan
