# Robust CIFAR-10 Classification from 500 Images

**Course:** Deep Learning for Perception, Assignment 2 (Section E), Fall 2026
**Author:** Amama (23F-6014), FAST NUCES

Train a small CNN from scratch on only 500 CIFAR-10 images (400 train / 100 validation) and make it hold up on a **shifted** test set (rotation, brightness/contrast change, blur, noise, a hidden colour corruption and a random occlusion).

## Repository contents

| File | Description |
|---|---|
| `DL_ASS02_23F_6014.ipynb` | Complete, executed notebook (Parts A to F) with all outputs and plots |
| `Assignment2_Report.docx` | One-page results summary |
| `README.md` | This file |

## Results

Shifted-test accuracy of the final model: **28.77% ± 0.99%** (mean ± std over seeds 0, 1, 2). Clean-test accuracy: 41.40% ± 2.21%.

### Baseline (Part A, no augmentation)

| Train | Validation | Clean test | Shifted test |
|---|---|---|---|
| 64.75 ± 2.70 | 40.00 ± 0.82 | 38.52 ± 0.27 | 17.82 ± 0.82 |

The baseline overfits (train much higher than validation) and fails under covariate shift (clean-test to shifted-test drop of about 21 points).

### Ablation (Part D, mean ± std over 3 seeds, accuracy in %)

| Configuration | Val | Clean test | Shifted test |
|---|---|---|---|
| No augmentation | 41.67 ± 0.94 | 40.74 ± 1.74 | 17.34 ± 0.43 |
| Geometric only | 38.67 ± 3.30 | 38.74 ± 1.30 | 17.09 ± 0.69 |
| Photometric only | 41.00 ± 4.55 | 40.58 ± 2.93 | 18.58 ± 0.41 |
| Geometric + photometric | 45.33 ± 1.70 | 43.99 ± 0.13 | 21.91 ± 0.96 |
| + Cutout | 43.00 ± 1.63 | 41.40 ± 2.21 | **28.77 ± 0.99** |
| + Mixup + label smoothing | 44.00 ± 0.82 | 40.96 ± 1.54 | 27.49 ± 1.05 |
| Final model (+ Cutout) | 43.00 ± 1.63 | 41.40 ± 2.21 | **28.77 ± 0.99** |

The largest single gain came from **Cutout** (+6.86 points over geometric + photometric), which matches the random rectangular occlusion in the shifted test set.

### Other parts

- **Part E (error analysis):** per-class accuracy, normalized confusion matrix, and 8 misclassified images are in the notebook. Bird and cat are weakest even on clean images; horse loses the most under the shift. On the hidden-corruption-only tests: grayscale 36.14%, channel swap 32.72%, inverted colours (never trained on) 12.66%.
- **Part F (test-time augmentation, bonus):** averaging softmax over 6 copies (original, horizontal flip, four 2-pixel shifts) changed shifted accuracy from 28.77% to 28.85% (+0.08 points) at about 6.1x the inference time, so it was not worthwhile.

## Final model

- **Architecture** (189,474 trainable parameters, limit 1,000,000): two 3×3 convs (32, 64 filters), an Inception-style module (1×1, 3×3, 5×5 and pooling branches with 1×1 reductions, 120 channels), and a 3×3 conv (128 filters), each followed by BatchNorm, then Global Average Pooling, Dropout (0.3) and a 10-way softmax.
- **Augmentation** (on the fly, training data only): rotation ±25°, 10% shift and zoom, horizontal flip, brightness and contrast jitter, Gaussian blur (30% of images), Gaussian noise, random grayscale and channel swap (20% each), Cutout (16×16 black square on 50% of images).
- **Training:** Adam (lr 1e-3), batch size 32, max 60 epochs, L2 weight decay 1e-5, `ReduceLROnPlateau` and `EarlyStopping` on validation loss. No pretrained weights.

## How to reproduce

### 1. Environment

Python 3.10+ is recommended.

```bash
pip install tensorflow numpy scipy scikit-learn matplotlib jupyter
```

The notebook was developed with Keras 3 (TensorFlow 2.16+) on CPU. Other versions may give slightly different numbers.

### 2. Run

```bash
jupyter notebook DL_ASS02_23F_6014.ipynb
```

Choose **Kernel > Restart & Run All**. CIFAR-10 is downloaded automatically by `keras.datasets.cifar10`; no manual data download is needed. Everything is run once per seed (0, 1, 2), so the full notebook takes a while on CPU.

### 3. Seeds (all fixed in the notebook)

| Purpose | Seed |
|---|---|
| Model / training runs | 0, 1, 2 |
| Selecting the 500 training images (50 per class) | 123 |
| 400 / 100 train-validation split (stratified) | 42 |
| Shifted test set generation | 999 |

`tf.config.experimental.enable_op_determinism()` is enabled. Exact numbers can still vary across TensorFlow versions and hardware.

### 4. Generated files

Part D saves the three final models as `final_model_seed0.keras`, `final_model_seed1.keras` and `final_model_seed2.keras`, which Parts E and F reuse. The shifted test set is generated inside the notebook, not stored.

## Data and rules followed

- `train_small`: 500 CIFAR-10 training images (50 per class), split 400 train / 100 validation.
- `clean_test`: the original 10,000 CIFAR-10 test images (reporting only).
- `shifted_test`: the 10,000 test images passed through a seeded corruption pipeline: rotation up to ±25°, brightness/contrast change, blur, Gaussian noise, random grayscale or channel swap (the hidden corruption), and a 16×16 black occlusion (25% of the image).
- No pretrained weights, at most 1,000,000 trainable parameters, at least one Inception-style module.
- Augmentation is applied on the fly to training data only; validation data is never augmented.
- Hyperparameters were set using validation data only. Maximum 60 epochs, 3 seeds, mean ± std reported.

## Limitations

- The final configuration (Cutout) was chosen by design reasoning because the validation accuracies of the last three configurations (45.33%, 43.00%, 44.00%) were too close to separate on 100 clean images. The shifted-test results had already been displayed by then, so the choice is not fully independent of the test data.
- The hidden corruption was written by the author (grayscale or channel swap), and the training augmentation includes both, so Part E4 is not a test on a truly unseen corruption. Inverted colours (never trained on) is the closest check of that, and accuracy falls to 12.66%.
