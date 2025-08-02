# GINGER
GINGER: A reproducible deep learning pipeline in Python and R for taxonomic analysis. This tool trains a model on your image data, extracts features, and runs a full suite of statistical analyses (PCA, UMAP, SHAP) to aid in species discovery and validation.
# GINGER: The Genomic and Image Networked Grouping Explorer

**Version:** 3.1 (Dual Python/R Pipelines)
**Computational Pipeline Developed By:** Anima Audire Labs, LLC, with assistance from Gemini

---

## 1. Abstract

The **GINGER** pipeline is a robust and reproducible bioinformatics tool designed for the taxonomic analysis of species with high phenotypic plasticity. It provides a state-of-the-art, end-to-end deep learning workflow that integrates image processing, model training, feature extraction, and a comprehensive suite of statistical analyses. This project provides parallel pipelines in both **Python** and **R** that are controlled by the same simple configuration file, making these advanced techniques accessible to researchers from diverse computational backgrounds.

---

## 2. The GINGER Pipeline Workflow

The pipeline is architected as a single script (either `GINGER_pipeline.py` or `GINGER_pipeline.R`) controlled by a **master configuration file (`config.yaml`)**. The user can choose to run any or all of the three internal stages:

1.  **Model Training:** Trains a deep learning model to recognize the visual differences between species using transfer learning.
2.  **Deep Feature Extraction:** Uses the trained model to convert every image into a rich numerical "fingerprint."
3.  **Visualization & Analysis:** Analyzes the "digital fingerprints" to find patterns, create plots (PCA, UMAP, etc.), and run statistical tests.

---

## 3. Installation and Setup

Choose the setup instructions for your preferred language.

### Option 1: Python Pipeline Setup

**Prerequisites:**
* **Python 3.10+ (64-bit)**

**Setup Steps:**

1.  **Navigate to Project Directory:**
    ```powershell
    cd C:\path\to\your\GINGER_pipeline
    ```
2.  **Create & Activate Virtual Environment:**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
3.  **Install All Python Packages from `requirements.txt`:**
    ```powershell
    pip install -r requirements.txt
    ```

### Option 2: R Pipeline Setup

**Prerequisites:**
* **R** and **RStudio Desktop**

**Setup Steps:**

1.  **Install Core R Packages**: Open RStudio and run the following command in the Console.
    ```R
    install.packages(c("torch", "torchvision", "magrittr", "yaml", "logging", "argparse", "rsample", "dplyr", "readr", "stringr", "ggplot2", "tidyr", "R.utils", "MASS", "umap", "dbscan", "randomForest", "plotly", "magick"))
    ```
2.  **Install Torch Backend**: After the packages are installed, run this second command in the RStudio Console.
    ```R
    torch::install_torch()
    ```

---

## 4. How to Run the Pipeline

1.  **Create Your Configuration:**
    * Copy `config.example.yaml` and rename it to `config.yaml`.
    * Open and edit your new `config.yaml`. **Crucially, update the `image_data_root` and `output_dir` paths.**

2.  **Run from the Terminal:**
    * **To run the Python pipeline:**
        ```powershell
        python GINGER_pipeline.py --train --extract --visualize
        ```
    * **To run the R pipeline:**
        ```powershell
        Rscript GINGER_pipeline.R --train --extract --visualize
        ```
---
## 5. Broader Applications in a University Setting

While designed for *Hexastylis*, the GINGER pipeline is a versatile tool with numerous innovative applications across scientific and university domains.

* **Taxonomy & Systematics**
    * **Scientific Use:** Rapidly screen thousands of specimens from digitized herbarium collections to flag potential new species, identify mislabeled sheets based on outlier detection, or delineate cryptic species complexes that are morphologically similar but genetically distinct.
    * **Non-Technical Analogy:** Think of it as a super-powered museum assistant. It can scan an entire collection of pressed plants and instantly flag any that look out of place or might be a new species no one has noticed before.

* **Ecology & Phenotyping**
    * **Scientific Use:** Quantify morphological variation across environmental gradients. For example, analyze leaf shape changes in a plant species at different elevations, moisture levels, or soil types to study phenotypic plasticity.
    * **Non-Technical Analogy:** It can measure how a plant's appearance changes based on where it lives. For example, it can show if the same type of tree grows leaves that are rounder at the bottom of a mountain and pointier at the top.

* **Agriculture & Crop Science**
    * **Scientific Use:** Analyze images of crops to detect early signs of nutrient deficiency or disease based on subtle color and texture changes. It can also be used to phenotype different cultivars by quantifying differences in fruit size, shape, or plant architecture.
    * **Non-Technical Analogy:** It's like a "robot farmer" that can look at pictures of a field of corn and spot sick plants weeks before a human could. It can also help breeders by precisely measuring which new crop varieties grow the biggest and best fruit.

* **Cell Biology & Medicine**
    * **Scientific Use:** Classify cell types, disease states (e.g., cancerous vs. healthy), or the effects of a drug treatment from microscope slide images. The feature extraction can turn complex cell images into quantitative data for statistical analysis.
    * **Non-Technical Analogy:** It can be trained to be a digital pathologist. It looks at cells under a microscope and learns to tell the difference between healthy ones and cancerous ones, helping doctors make faster diagnoses.

* **Educational Tool**
    * **Scientific Use:** Serve as a complete, real-world example for teaching students in bioinformatics, data science, or machine learning courses how to build and use an end-to-end AI research pipeline, from data ingestion to final publication-ready figures.
    * **Non-Technical Analogy:** It's a perfect "science fair project on steroids." It provides a ready-made example that lets students learn how real-world AI research is done, using the same tools that professional scientists use.

---
## 6. Citing the GINGER Pipeline

If you use this software in your research, please use the "Cite this repository" button on the GitHub sidebar.
