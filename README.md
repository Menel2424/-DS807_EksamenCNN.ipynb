# DS807 Exam – CNN (Problem 1, Questions 2 & 3)

This repository contains **two versions** of the same exam assignment (Problem 1, questions 2 and 3, DS807), each using a different pre-trained CNN backbone for transfer learning on the EuroSAT satellite image dataset:

| File | Backbone | Notes |
|---|---|---|
| `DS807_EksamenCNN_ResNet50.ipynb` | **ResNet50** | Preprocessing done explicitly with `preprocess_input` before training; VAE trained via a custom `train_step`; robustness tested against both Gaussian noise **and** blur. |
| `DS807_CNN_P1_Q2_Q3-2.ipynb` | **EfficientNetB0** | Resizing and preprocessing done *inside* the model graph; VAE loss added via a custom `add_loss` layer; robustness tested against Gaussian noise only, with a more detailed sweep (11 noise levels). |

Both notebooks answer the same three sub-questions and follow the same overall pipeline, so this README describes the shared structure once and calls out the differences between the two versions where relevant.

## Assignment overview

Both notebooks cover:

1. **Question 2 – Classification model (CNN)**: Build, train, fine-tune, and evaluate a transfer-learning image classifier on EuroSAT.
2. **Question 2.3 – Latent space investigation**: Compare a supervised approach (PCA on CNN features) with an unsupervised approach (a Variational Autoencoder) for reducing the data to a 2D latent space.
3. **Question 3 – Model interpretability and robustness**: Use Grad-CAM to visualize what the model focuses on, and test how classification accuracy degrades under image corruption (noise / blur).

### Data
- **Dataset**: `nielsr/eurosat-demo`, loaded via Hugging Face `datasets`.
- **Classes** (10): `industrial_buildings`, `residential_buildings`, `annual_crop`, `permanent_crop`, `river`, `sea_lake`, `herbaceous_vegetation`, `highway`, `pasture`, `forest`.
- **Split**: 64% train / 16% validation / 20% test (stratified by class, `random_state=42`) in both notebooks.

### Question 2 – Classification model
- Both notebooks use **transfer learning**: a frozen, ImageNet-pretrained backbone, a `GlobalAveragePooling2D` layer, a `Dense(128)` "feature vector" layer, `Dropout(0.3)`, and a softmax output layer over the 10 classes.
- **Two-stage training** in both:
  1. **Baseline**: backbone fully frozen, only the new head is trained.
  2. **Fine-tuning**: the last 30 layers of the backbone are unfrozen and trained further at a lower learning rate.
- **Callbacks**: `EarlyStopping` (patience 5, restores best weights) and `ModelCheckpoint` (saves best model by `val_loss`) in both.
- **Evaluation**: test accuracy/loss, classification report, confusion matrix, and loss/accuracy curves for both training stages, in both notebooks.
- **Key difference**: the ResNet50 version explicitly preprocesses images with `preprocess_input` before feeding the model; the EfficientNetB0 version resizes and preprocesses images *inside* the model (as part of the computation graph), so raw images are passed directly to `.fit()`.

### Question 2.3 – PCA vs. VAE
- **Approach A (supervised)**: Features are extracted from the trained CNN's `feature_vector` layer and reduced to 2D with PCA; the result is visualized per class for train and validation data. Identical logic in both notebooks.
- **Approach B (unsupervised)**: A convolutional Variational Autoencoder (encoder/decoder architecture, 2D latent space) is trained from scratch directly on the raw images, without using any labels. The resulting latent space is then visualized per class (for comparison only — the VAE itself never sees the labels).
- **Key difference**: the ResNet50 notebook implements the VAE as a custom `tf.keras.Model` subclass with a hand-written `train_step`/`test_step` that computes reconstruction loss + KL divergence. The EfficientNetB0 notebook instead wraps the reconstruction + KL loss computation in a custom `VAELossLayer` and adds it to the model via `add_loss`, letting the standard `.fit()` / `.compile()` handle training without a custom loop.

### Question 3 – Interpretability and robustness
- **Grad-CAM**: Both notebooks implement Grad-CAM to visualize which regions of an image drive the model's prediction, illustrated on the `highway` and `river` classes.
  - The ResNet50 version builds separate sub-models to extract intermediate outputs and computes gradients with respect to the last convolutional block (`conv5_block3_out`) of the ResNet50 backbone.
  - The EfficientNetB0 version instead rebuilds the entire forward pass (resizing → preprocessing → augmentation → backbone → classification head) as a single computation graph in one function (`compute_gradcam`), specifically to avoid disconnected-gradient issues that can occur when re-using pre-built sub-models.
- **Robustness testing**:
  - The ResNet50 version tests robustness against **both Gaussian noise** (σ = 0, 15, 30, 45, 60) **and Gaussian blur** (σ = 0, 1, 2, 3, 4), tracking both overall and per-class accuracy, and reports the most fragile classes for each corruption type.
  - The EfficientNetB0 version tests robustness against **Gaussian noise only**, but with a finer sweep across 11 noise levels (σ from 0 to 50), plotting overall test accuracy against noise level.

## Requirements

Both notebooks are written to run in **Google Colab** (they use `google.colab.drive` to save trained models to Google Drive). Required packages:

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

## How to run

For **either** notebook:

1. Open it in Google Colab (recommended) or Jupyter.
2. Run the **Question 2** cell(s) first — this mounts Google Drive, loads and splits the data, and builds/trains/evaluates the baseline and fine-tuned classification model.
3. Run the **Question 2.3** cell(s) next — this depends on the trained model and data variables from step 2, and produces the PCA and VAE latent-space visualizations.
4. Run the **Question 3** cell(s) last — this depends on the trained model and test data from step 2, and produces the Grad-CAM visualizations and the robustness (noise/blur) plots.

> **Note**: Cells must be run in order within each notebook — later sections depend on variables (the trained model, data splits, class names) created in earlier sections.

## Saved models

Both notebooks save the best model weights (by lowest `val_loss`) to Google Drive at `/content/drive/MyDrive/models`:

- ResNet50 version: `resnet50_baseline.keras`, `resnet50_finetuned.keras`
- EfficientNetB0 version: `EFFICIENTNETB0_baseline.keras`, `EFFICIENTNETB0_finetuned.keras`

## Output / results

Both notebooks produce:
- Sample training images with class labels
- Model architecture summary
- Training/validation loss and accuracy curves for the baseline and fine-tuned model
- Classification report and confusion matrix on the test set
- 2D latent-space visualizations comparing supervised PCA vs. unsupervised VAE
- Grad-CAM heatmaps for selected classes (`highway`, `river`)
- Robustness plots showing how test accuracy degrades under image corruption (noise in both; noise + blur in the ResNet50 version)
