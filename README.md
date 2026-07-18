# 🌿 Plant Disease Classifier

A deep learning image classifier that identifies plant diseases from leaf photos, built with PyTorch (MobileNetV2 transfer learning) and deployed as an interactive Streamlit web app.

## Demo

Upload a leaf image and the model predicts the disease (or healthy status) across 38 classes spanning 14 crop species (Apple, Corn, Grape, Tomato, Potato, and more).

**Live app:** _[add your Streamlit Cloud link here after deployment]_

## Overview

- **Task:** Multi-class image classification (38 classes)
- **Framework:** PyTorch
- **Architecture:** MobileNetV2 (pretrained on ImageNet), fine-tuned on plant disease images
- **Dataset:** [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) — an augmented version of PlantVillage, ~87K images across 38 classes
- **Training environment:** Google Colab (T4 GPU)

## Approach

1. Loaded MobileNetV2 pretrained on ImageNet, froze the base convolutional layers
2. Replaced the final classifier layer to output 38 classes, trained just this head first
3. Fine-tuned the last few convolutional blocks with a low learning rate (1e-5) to adapt pretrained features to leaf images
4. Evaluated using precision, recall, and F1-score per class (not just overall accuracy), given class-imbalance risk in image datasets

## Results

- **Overall validation accuracy: 99%** (17,572 validation images)
- Macro avg and weighted avg F1-score: 0.99
- Nearly all classes scored 0.95–1.00 across precision/recall/F1

**Weakest classes (still strong, but worth noting):**
| Class | Precision | Recall |
|---|---|---|
| Corn — Cercospora Gray Leaf Spot | 0.98 | 0.92 |
| Corn — Northern Leaf Blight | 0.94 | 0.98 |
| Tomato — Late Blight | 0.93 | 0.99 |
| Tomato — Target Spot | 0.98 | 0.92 |

The confusion is concentrated between diseases *within the same crop* that share visually similar lesion patterns (e.g. two corn diseases, two tomato diseases) — rather than random errors across unrelated classes. This suggests the model is learning genuine visual disease features rather than memorizing noise.

## Limitations (tested manually, not just theoretical)

This model was trained on PlantVillage-style images: single leaf, isolated, plain background, consistent lighting. Real-world testing exposed a real distribution-shift limitation:

- **Out-of-distribution plant (not in training set):** confidence dropped to ~56%, with inconsistent, scattered top-3 predictions — expected, since the model has no "unknown" rejection class and must always pick from its 38 known classes.
- **Correct crop (tomato), but natural/cluttered background with multiple leaves in frame:** model misclassified the leaf entirely (predicted a different crop), with confidence dropping to 65%, despite 98%+ accuracy on the clean validation set.
- **Correct crop, natural background, but closer single-leaf framing:** model correctly predicted the disease at 99.7% confidence.

**Takeaway:** validation accuracy on PlantVillage does not fully represent real-field performance. A production-ready version would need training on more diverse, real-world field images (e.g. the [PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset)) or stronger background/augmentation diversity during training.

## Tech Stack

- Python, PyTorch, torchvision
- scikit-learn (evaluation metrics)
- Streamlit (deployment/UI)
- Google Colab (training)

## Project Structure

```
├── app.py                      # Streamlit web app
├── plant_disease_model.pth     # Trained model weights
├── training_notebook.ipynb     # Colab training + evaluation notebook
├── requirements.txt            # Python dependencies
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

- Add an "unknown/out-of-distribution" rejection mechanism
- Train/fine-tune on real-field images (e.g. PlantDoc) to close the distribution-shift gap
- Add Grad-CAM visualization to show which part of the leaf the model is focusing on