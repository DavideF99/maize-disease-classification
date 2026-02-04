# Large-scale Maize Disease Classification with Hierarchical Ensembles

[![Deep Learning](https://img.shields.io/badge/Model-Hierarchical--Ensemble-blue.svg)](https://github.com/DavideF99/maize-disease-classification)
[![PyTorch Lightning](https://img.shields.io/badge/Framework-PyTorch--Lightning-orange.svg)](https://www.pytorchlightning.ai/)

A professional model card and project overview for a multi-label maize disease classification system. This project implements a hierarchical ensemble architecture designed to handle severe class imbalance and achieve high-precision results for agricultural leaf disease detection.

---

## 📄 Model Card

### Project Description
This project focuses on the precise identification of various maize foliar diseases from high-resolution leaf images. The system is built using a hierarchical ensemble approach, separating "Healthy" leaves from "Diseased" ones before further classification by specialized models.

### Dataset
- **Source**: Images were sourced from a dataset hosted on a South African University's website, featuring high-resolution images of maize leaves in various agricultural conditions.
- **Task**: Multi-label classification (Single images can exhibit multiple disease symptoms).
- **Classes**:
  1. Grey Leaf Spot (GLS)
  2. Northern Corn Leaf Blight (NCLB)
  3. Phaeosphaeria Leaf Spot (PLS)
  4. Common Rust (CR)
  5. Southern Rust (SR)
  6. Healthy (No Foliar Symptoms)
  7. Other (Leaves with tearing, insects, etc.)
  8. Unidentified (Diseases not belonging to the above)
- **Class Imbalance**: The dataset exhibits a significant imbalance, with a distribution of approximately **2070** healthy/majority class images versus **285** diseased/minority class images.

### Architecture: Hierarchical Ensemble Logic
The final model architecture follows a "Gatekeeper & Specialist" hierarchical pattern to maximize precision:

1.  **Gatekeeper (EfficientNet-B0)**: A binary classifier trained specifically to distinguish if a leaf is healthy or shows *any* signs of disease. This model acts as a high-recall filter.
2.  **Hero Model (Specialist - EfficientNet-B0)**: If the Gatekeeper detects disease, the image is passed to the Specialist. This model is fine-tuned for multi-label classification across all disease categories.
3.  **Inference Strategy**:
    - **Test-Time Augmentation (TTA)**: Multiple viewpoints (Original, Horizontal Flip, Vertical Flip) are averaged to ensure robust predictions.
    - **Dynamic Thresholding**: Optimal probability thresholds are calculated per-class during validation to balance Precision and Recall for rare diseases like Southern Rust.

---

## 📈 Iterative Development Process
The project evolved through multiple stages to arrive at the hierarchical ensemble:

1.  **`mobilenet`**: Initial baseline experiments using MobileNetV2. Quick to train but lacked the feature extraction depth for subtle disease patterns.
2.  **`efficientnet`**: Migrated to EfficientNet-B0 as the primary backbone. The compound scaling provided a significantly better performance-to-parameters ratio.
3.  **`efficientnet_optimized`**: Integrated **Optuna** for hyperparameter optimization (Learning Rate, Dropout, Weight Decay) and implemented fine-tuning of the final EfficientNet feature blocks.
4.  **`ensemble`**: Transitioned to the **Hierarchical Ensemble** (Gatekeeper + Hero) architecture to solve the "False Positive" issue where healthy leaves were occasionally misclassified as diseased.
5.  **`batch-process`**: Final optimizations for large-scale inference and deployment, ensuring the model can handle production-level data throughput.

---

## ⚠️ Ethics & Risks

> [!CAUTION]
> **NOT FOR FIELD DECISION-MAKING**
> This model is a research tool and should **not** be used for making actual farming decisions or applying chemical treatments in a real-world agricultural setting.

- **Risk of Misidentification**: While high in precision, the model can still misidentify diseases. Incorrect diagnosis could lead to improper pesticide application, harming crops or the environment.
- **Dataset Bias**: The model was trained on data from specific South African regions. Its performance on maize varieties or agricultural conditions in other parts of the world has not been validated.
- **Inference Latency**: The ensemble approach increases computational requirements; however, this is compensated for by the use of EfficientNet-B0 backbones.

---

## 🛠️ How to Use

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Running Inference
The system provides a Batch Prediction interface:
```bash
python src/TVT/batch_predict.py --input_dir data/test_images/ --output_csv predictions.csv
```

### Web Interface
A Gradio-based web interface is available for interactive testing:
```bash
python src/app.py
```

---

## 🏁 Final Model Justification
We settled on the **Gatekeeper + Specialist Ensemble** because it effectively decouples the "Is it sick?" question from "What does it have?". This architecture, combined with **EfficientNet-B0**, **TTA**, and **Dynamic Thresholds**, provided the most robust performance across the severely imbalanced dataset, ensuring that "Healthy" leaves are filtered out with 95%+ F1-score before disease-specific classification begins.
