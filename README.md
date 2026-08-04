CNN
# DS807 Exam – CNN (Problem 1)

This repository contains a Jupyter notebook (`DS807_EksamenCNN-4.ipynb`) with the answer to Problem 1, questions 2 and 3, for the DS807 course. The assignment covers image classification of satellite images (EuroSAT) using a CNN-based model (ResNet50), followed by an analysis of the model's interpretability and robustness.

## Contents

The notebook consists of two main sections:

### Question 2 – Classification model (CNN)
- **Data**: The dataset `nielsr/eurosat-demo` is downloaded via Hugging Face `datasets` and contains satellite images split across 10 classes:
  `industrial_buildings`, `residential_buildings`, `annual_crop`, `permanent_crop`, `river`, `sea_lake`, `herbaceous_vegetation`, `highway`, `pasture`, `forest`.
- **Data split**: 64% train / 16% validation / 20% test (stratified by class).
- **Model**: Transfer learning using a `ResNet50` (ImageNet weights) as a feature extractor, followed by `GlobalAveragePooling2D`, a `Dense(128)` layer (`feature_vector`), `Dropout`, and a softmax output layer.
- **Two-stage training**:
  1. **Baseline**: All ResNet50 layers are frozen; only the new classification layers are trained.
  2. **Fine-tuning**: The last 30 layers of ResNet50 are unfrozen and trained further with a lower learning rate.
- **Evaluation**: Test accuracy/loss, classification report, and confusion matrix.
- **Latent space analysis (PCA vs. VAE)**:
  - **Approach A**: Features from the CNN's `feature_vector` layer are reduced to 2D using PCA and visualized per class.
  - **Approach B**: A separate Variational Autoencoder (VAE) is trained from scratch (unsupervised) on the images, and the resulting 2-dimensional latent space is visualized in the same way.
- **Training history**: Loss and accuracy curves for both the baseline and fine-tuned model.

### Question 3 – Model interpretability and robustness
- **Grad-CAM**: Heatmaps are generated to show which parts of the image the model focuses on, illustrated for the `highway` and `river` classes.
- **Robustness testing**:
  - **Gaussian noise** is added to the test images at increasing strength (σ = 0, 15, 30, 45, 60), and accuracy is measured overall and per class.
  - **Gaussian blur** is added similarly at increasing strength (σ = 0, 1, 2, 3, 4).
  - Results are plotted, and the classes most vulnerable to noise/blur are identified.

## Requirements

The notebook is written to run in **Google Colab** (it uses `google.colab.drive` to save models to Google Drive). The following packages must be installed:

```
tensorflow
numpy
matplotlib
scikit-learn
seaborn
datasets   # Hugging Face
```

Installation (if not using Colab):
```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn datasets
```

## How to run the notebook

1. Open `DS807_EksamenCNN-4.ipynb` in Google Colab (recommended) or Jupyter.
2. Run the cells under **Problem 1 question 2** in order:
   - Mounts Google Drive (Colab only) and creates the folder `/content/drive/MyDrive/models` to save model checkpoints.
   - Downloads and prepares the data.
   - Builds, trains, and evaluates the baseline and fine-tuned ResNet50 model.
   - Runs the PCA and VAE latent space analysis.
3. Then run the cells under **Problem 1 Question 3**:
   - Requires `model_baseline`, `X_test`, `X_test_processed`, `y_test`, and `class_names` from question 2 to be present in the same running session.
   - Generates Grad-CAM visualizations and robustness plots for noise and blur.

> **Note**: Run the cells in the order they appear in the notebook — question 3 depends on variables and the trained model from question 2.

## Saved models

During training, the best model weights (based on lowest `val_loss`) are automatically saved to:
- `resnet50_baseline.keras`
- `resnet50_finetuned.keras`

in the folder `/content/drive/MyDrive/models` (requires Google Drive access in Colab).

## Output / results

The notebook produces, among other things:
- Sample images from the training set with class names
- Model architecture summary
- Training and validation curves (loss/accuracy) for both models
- Classification report and confusion matrix on the test set
- 2D visualizations of the latent space (PCA vs. VAE)
- Grad-CAM heatmaps for selected classes (highway, river)
- Graphs of accuracy drop under increasing noise and blur, plus an overview of the most vulnerable classes
