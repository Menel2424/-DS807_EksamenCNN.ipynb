# DS807 – Problem 1: Land Use and Land Cover Classification (Deep Learning)

This repo contains two notebooks answering **Question 2** and **Question 3** of Problem 1
("Land Use and Land Cover Classification") from the DS807 exam (Winter 25/26). The task is to
classify satellite images from the **EuroSAT** dataset (10 classes, e.g. Forest, Highway,
Residential, River, SeaLake, etc.) using CNNs, and then analyze and interpret the models.

## File overview

| File | Content | Answers |
|---|---|---|
| `DS807_P1_Q2_Q3_CNN.ipynb` | EfficientNetB0 model (224×224), latent space analysis (PCA vs. VAE), Grad-CAM and robustness test | Question 2.1, 2.3 and 3 |
| `DS807_P1_Q2_Efficientnet.ipynb` | Comparison of EfficientNetB0 (64×64) vs. EfficientNetB5 (224×224) | Question 2.2 (model scaling) |

Both notebooks are written to run in **Google Colab** (they use `google.colab.drive` to save
models to Google Drive).

---

## Data

- **Source:** EuroSAT (RGB) loaded via Hugging Face `datasets`: `load_dataset("nielsr/eurosat-demo")`.
- **Classes (10):** `industrial_buildings, residential_buildings, annual_crop, permanent_crop,
  river, sea_lake, herbaceous_vegetation, highway, pasture, forest`.
- **Split:** Data is split in two steps using `train_test_split` (stratified on label,
  `random_state=42`):
  1. 80 % train+val / 20 % test
  2. Of the 80 %: 80 % train / 20 % val

  Result: **64 % train / 16 % validation / 20 % test**.
- Images are originally **64×64×3**. Resizing to the model's expected input size (e.g. 224×224 for
  EfficientNetB0/B5) happens **inside the model** (`tf.keras.layers.Resizing`), so the raw data
  stays at 64×64 and augmentation/normalization is handled consistently across train/val/test.

---

## `DS807_P1_Q2_Q3_CNN.ipynb`

### Part 1 – Question 2.1: CNN training and model selection
- **Architecture:** Transfer learning with **EfficientNetB0** (ImageNet weights), input resized to 224×224.
- **Data augmentation:** Random horizontal flip, rotation (±5 %), zoom (±10 %).
- **Two-stage training:**
  1. **Baseline:** Base model frozen, only the new classification head is trained (Dense 128 →
     Dropout 0.3 → softmax over 10 classes). Adam, lr = 1e-4, up to 50 epochs.
  2. **Fine-tuning:** The last 30 layers of EfficientNetB0 are unfrozen and trained further. Adam,
     lr = 1e-4, up to 15 epochs.
- **Model selection:** `EarlyStopping` (monitor = `val_loss`, patience = 5,
  `restore_best_weights=True`) combined with `ModelCheckpoint` (saves the best model based on
  `val_loss`) — i.e. minimum validation loss is used as the stopping criterion/checkpoint metric,
  not maximum validation accuracy.
- **Evaluation:** Test accuracy/loss, `classification_report` (precision/recall/F1 per class) and
  confusion matrix (heatmap) on the test set.

### Part 2 – Question 2.3: Latent Space (PCA vs. VAE)
- **Approach A (supervised):** Features are extracted from the trained CNN's penultimate layer
  (`feature_vector`, 128-dim), reduced to 2D with PCA, and plotted colored by class (train + val).
- **Approach B (unsupervised):** A **Variational Autoencoder (VAE)** is built and trained from
  scratch using only the images (labels are ignored):
  - Encoder: 3× Conv2D (32/64/128 filters, stride 2) → Dense(256) → `z_mean`/`z_log_var` (latent
    dim = 2) → sampling layer (reparameterization trick).
  - Decoder: Dense → reshape → 3× Conv2DTranspose (upsampling) → sigmoid output.
  - Loss: reconstruction loss (binary crossentropy) + KL divergence, implemented via a custom
    `VAELossLayer` (`add_loss`).
  - Trained for 30 epochs on normalized images (0–1).
  - The 2D latent representations (`z_mean`) for train/val are plotted colored by class, so they
    can be directly compared with the PCA plot from Approach A.

### Part 3 – Question 3: Interpretability and robustness
- **Grad-CAM:** Custom implementation (`compute_gradcam`) that runs a single forward pass through
  resizing → preprocessing → augmentation → EfficientNetB0 backbone → pooling → classification
  head, and uses `GradientTape` to compute gradients of the class score with respect to the
  feature maps. Heatmaps are overlaid on the original image for examples from the **River** and
  **Highway** classes.
- **Robustness test (noise):** Gaussian noise is added to the test images' pixel values with
  increasing standard deviation (σ from 0 to 50 in 11 steps), and the model's accuracy is measured
  at each noise level. The result is plotted as accuracy vs. σ to investigate whether the model
  degrades gradually or collapses suddenly.

---

## `DS807_P1_Q2_Efficientnet.ipynb`

Answers **Question 2.2 (Model Scaling)** by comparing two EfficientNet variants that differ in
both depth/width (model size) and resolution:

| | Model | Input resolution | Baseline optimizer | Fine-tune optimizer |
|---|---|---|---|---|
| Cell 1 | **EfficientNetB0** | 64×64 | Adam, lr = 1e-3 | Adam, lr = 1e-4 |
| Cell 2 | **EfficientNetB5** | 224×224 | Adam, lr = 1e-3 | Adam, lr = 1e-4 |

Both follow the same recipe as in the CNN notebook (frozen base → 15 epochs of fine-tuning on the
last 30 layers, same augmentation, same `EarlyStopping`/`ModelCheckpoint` strategy, same
evaluation with classification report and confusion matrix). The purpose is to discuss the
trade-off between model size/resolution (parameters, training time) and performance, with
reference to the EfficientNet paper's compound scaling principle (Tan & Le, 2019).

> **Note:** This file uses lr = 1e-3 in the baseline stage (vs. 1e-4 in `..._CNN.ipynb`), and in
> the B0 cell the input stays at 64×64 (no resizing to 224×224), while the B0 model in the CNN
> notebook is resized to 224×224. This is an intentional part of the scaling experiment
> (resolution as a variable), but is worth stating explicitly in the report so the methodology is
> transparent.

---

## Running the notebooks

1. Open the notebook in **Google Colab**.
2. Run the first cell — it will request access to Google Drive (`drive.mount`) and create
   `/content/drive/MyDrive/models`, where trained models (`.keras` files) are saved automatically
   via `ModelCheckpoint`.
3. Required packages (already in Colab): `tensorflow`, `numpy`, `matplotlib`, `scikit-learn`,
   `seaborn`, `datasets` (Hugging Face).
4. Run the cells in order — each markdown heading corresponds to an exam question/sub-question.

## Hardware note

Training EfficientNetB5 at 224×224 as well as the VAE is computationally heavy. Use a GPU runtime
in Colab (Runtime → Change runtime type → GPU). Batch size defaults to 32; reduce it if you run
into memory issues.
