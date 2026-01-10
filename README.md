<div align="center">
  <a href="https://colab.research.google.com/drive/1PfxW4tnS8saZxGFmQinCzEVXuDk3OlBf?usp=sharing" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab">
    <br>
    <strong>Click to open in Google Colab</strong>
  </a>
</div>
---

# Computer Vision for Food Composition Estimation

This notebook implements a complete deep learning pipeline for estimating the composition of food images, specifically predicting the proportion of confetti (`c`) and sausage (`s`) components in dish cover images. The project employs state-of-the-art computer vision techniques with convolutional neural networks to solve a regression problem with constrained outputs (c+s+bg=1, where bg represents background).

## Project Structure

The notebook is organized into modular blocks, each serving a distinct purpose in the machine learning pipeline:

| Block | Purpose | Key Technologies | Outputs |
|-------|---------|------------------|---------|
| **BLOCK 0** | Initial setup and data preparation | `pathlib`, `zipfile`, `pandas` | Directory structure, verified dataframes with image paths |
| **BLOCK 1** | Exploratory Data Analysis (EDA) | `matplotlib`, `numpy` | Statistical summaries, visualizations, baseline metrics |
| **BLOCK 2** | Strong baseline model with 5-fold CV | `timm`, `PyTorch`, `albumentations` | Trained ConvNeXt-Tiny models, cross-validation results |
| **BLOCK 3** | Enhanced training with EMA and MixUp | EMA, MixUp augmentation | Improved checkpoints with advanced regularization |
| **BLOCK 4** | Inference with ensemble and TTA | Test-Time Augmentation (TTA), ensemble methods | Final submission file with predictions |
| **Additional** | Stratified k-fold creation | `StratifiedKFold`, quantile binning | Balanced folds for improved validation |
| **Advanced** | High-resolution model training | `convnext_small`, gradient accumulation | Premium checkpoints for ensemble |

## Technical Implementation Details

### Data Representation
The problem is framed as a three-class probability distribution estimation:
- Class 0: Confetti proportion (`c`)
- Class 1: Sausage proportion (`s`) 
- Class 2: Background proportion (`bg = 1 - c - s`)

This formulation ensures the sum-to-one constraint is naturally enforced through softmax normalization.

### Model Architecture
The solution utilizes pre-trained ConvNeXt architectures from the `timm` library:
- **ConvNeXt-Tiny**: Primary baseline model (256×256 input)
- **ConvNeXt-Small**: Enhanced model with higher resolution (384×384 input)

Each model is modified to output three logits corresponding to the three probability classes.

### Loss Function
A hybrid loss function combining KL-divergence and MAE components: 
Loss = KL(F.softmax(pred), target_distribution) + λ * MAE(pred_cs, true_cs)

This approach leverages both distribution matching and direct regression benefits.

### Training Strategies
1. **5-Fold Cross Validation**: Ensures robust performance estimation
2. **Cosine Annealing with Warmup**: Optimized learning rate schedule
3. **EMA (Exponential Moving Average)**: Smooths model weights for better generalization
4. **MixUp Augmentation**: Creates virtual training samples via convex combinations
5. **Gradient Accumulation**: Enables larger effective batch sizes on memory-constrained hardware

### Inference Pipeline
1. **Model Ensemble**: Combines predictions from multiple folds and architectures
2. **Test-Time Augmentation (TTA)**: 8 augmentations per image (flips, rotations)
3. **Simplex Projection**: Ensures predicted probabilities satisfy c+s≤1 constraint
4. **Weighted Averaging**: Different model types can be assigned different weights

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/Third_Task_clean/screenshots/1.png" width="250" height="530"/>
</p>

**Figure 1: Project Directory Structure**  
Illustrates the comprehensive file organization including initial data archives (`train_images_covers.zip`, `test_images_covers.zip`), processed dataframes with image paths (`train_index.csv`, `test_index.csv`), model checkpoints from different training configurations, and final submission files. The structure supports reproducible experimentation with multiple model versions.

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/Third_Task_clean/screenshots/2.png" width="780" height="630"/>
</p>

**Figure 2: Dataset Statistics and Extreme Examples**  
Presents key statistical properties of the training targets including mean values (c=0.369, s=0.039), correlation analysis showing negative correlation between confetti and background components (-0.939), and visual examples of extreme cases. The top section shows images with highest confetti proportion (>0.53), while the bottom section displays cases with zero sausage content, highlighting dataset diversity.

<p align="center">
  <img src="https://github.com/Figrac0/ML-Marathon-Solutions/blob/Third_Task_clean/screenshots/3.png" width="870" height="530"/>
</p>

**Figure 3: ConvNeXt-Tiny Training Progress (Fold 1)**  
Demonstrates the optimization trajectory during 18 training epochs with cosine annealing schedule. Shows MAE reduction from 0.044 to 0.006 on validation data, with learning rate decreasing from 2e-4 to 1.92e-6. The model achieves convergence within 30 seconds per epoch, illustrating efficient training with automatic mixed precision (AMP).

## Performance Metrics

The baseline model achieves:
- **CV MAE (average)**: 0.00645
- **Individual fold performance**: 0.00590-0.00763
- **Significant improvement** over naive mean predictor (0.04795 MAE)

## Usage Instructions

1. **Initial Setup**: Run BLOCK 0 to unpack data and verify integrity
2. **EDA**: Execute BLOCK 1 to understand data characteristics
3. **Training**: Run BLOCK 2 for baseline models or advanced blocks for enhanced versions
4. **Inference**: Use BLOCK 4 to generate final predictions
5. **Submission**: Upload `submission.csv` or `submission_ideal.csv` for evaluation

## Key Innovations

1. **Soft-label formulation**: Treats proportion estimation as probability distribution prediction
2. **Stratified k-fold**: Uses 2D binning on both c and s targets for balanced splits
3. **Photometric-only augmentations**: Preserves spatial relationships critical for proportion estimation
4. **Simplex projection**: Mathematical guarantee of constraint satisfaction in predictions
5. **Modular design**: Each block can be independently modified or replaced

## Dependencies
```js
timm>=0.9.0
torch>=2.0.0
albumentations>=1.3.0
opencv-python-headless>=4.8.0
pandas>=2.0.0
matplotlib>=3.7.0
```
