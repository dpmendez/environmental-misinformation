# Climate Misinformation Detection 🌍
This repository presents a proof-of-concept NLP pipeline for detecting misinformation in climate and environmental news. It combines classical machine learning models and transformer-based classifiers to analyze climate-related claims, with a focus on evaluation rigor, uncertainty, and reproducibility, rather than deployment-ready fact-checking.
The project also includes a lightweight FastAPI web application that allows users to interactively test multiple trained models. Predictions from the app should be interpreted as guidelines, not ground truth, as large-scale, high-quality labeled data remains a key limitation in this domain.

## 🔍 Problem Statement
Misinformation around climate change, renewable energy, and conservation undermines public trust, policy adoption, and collective climate action.
This project aims to:

* Compare classical ML baselines against transformer-based classifiers
* Explore NLP techniques for environmental misinformation detection
* Provide a scalable and modular research pipeline for climate discourse analysis
* Support downstream applications in research, NGOs, and policy analysis

This work is intended as a research prototype rather than a production moderation or fact-checking system.

## 📊 Data
The project uses climate-domain fact-checking and news sources:
* Climate Fever dataset: Environment-specific fact-check articles
Source: https://huggingface.co/datasets/tdiggelm/climate_fever
* Science Feedback – Climate News: Scraped expert-reviewed climate news articles
Source: https://science.feedback.org/climate-feedback/

Data is preprocessed for label unification and into train / validation / test splits.

## ⚙️ Methods
### Preprocessing
* Text normalization and cleaning
* Tokenization
* Label encoding
* Dataset construction for sklearn and transformer pipelines

### Models
Classical (TF-IDF based):
* Logistic Regression
* Random Forest
* Linear SVC
* XGBoost
Transformer-based:
* distilbert-base-uncased
* bert-base-uncased
* electra-base-discriminator

### Training
* Implemented using PyTorch and Hugging Face Transformers
* Optimizer: AdamW with learning rate scheduling
* Class imbalance handled during training where applicable
* Evaluation Metrics
    * Balanced accuracy
    * Weighted F1-score
    * Precision and recall (misinformation / false class)
    * ROC curves and AUC

## 📁 Repository Structure
```
.
├── app/                      # FastAPI web application
│   ├── main.py               # API and routing logic
│   ├── models/               # Saved models + thresholds
│   └── templates/            # HTML interface
├── data/                     # Processed datasets and splits
├── notebooks/                # End-to-end training workflows
├── src/                      # Core training and evaluation logic
│   ├── models.py             # Training functions and classes
│   └── viz.py                # Metric & ROC plotting utilities
├── analysis/
│   └── results/
│       ├── metrics_summary.csv
│       ├── metrics_deltas.csv
├── eval.py               # Unified evaluation script
├── requirements.txt
└── README.md
```

### 📓 Notebooks (notebooks/)
The notebooks walk through the training pipeline:
* Exploratory Data Analysis
* Text preprocessing and dataset construction
* Baseline model training (TF-IDF + Logistic Regression / XGBoost)
* Transformer fine-tuning with Hugging Face
* Training curves and confusion matrices

⚠️ Important:
All models must be trained via the notebooks first. This saves model artifacts, predictions, and metadata locally, which are later required for evaluation and analysis.

### Core Code (src/)
The src/ directory contains modular, reusable code for:
* Data preprocessing
* Baseline ML pipelines
* Transformer training loops
* Metric computation and aggregation
* Visualization of results (metrics and ROC curves)

### 📈 Evaluation & Analysis (eval.py)
A standalone script to evaluate any saved model, whether sklearn-based or transformer-based.
Supports:

* Hugging Face models saved via save_pretrained
* Sklearn pipelines saved with joblib

Outputs:
* Accuracy, balanced accuracy
* Precision, recall, F1
* ROC curves and AUC
* Confusion matrices

Aggregated Results:
Instead of hard-coded tables, model performance is summarized in
```analysis/results/metrics_summary.csv```
```analysis/results/metrics_deltas.csv```

These files compare all trained models consistently, highlight performance deltas between out-of-the-box and thresholded models, and serve as the source of truth for reported results.

### Plotting utilities (viz.py) generate:
* Metric comparison plots
* ROC curves across models

## 🚀 Results (Summary)
Rather than presenting fixed numbers in the README, all results are derived from the CSV files in analysis/results/.
Across experiments:

* Classical models remain competitive but transformer models outperform classical baselines on AUC and balanced accuracy
* High-confidence predictions are limited, reinforcing the need for cautious interpretation

The top-performing transformer and classical models, based on ROC-AUC, balanced accuracy and F1 for the positive class, are published on Hugging Face under my account <dpmendez>.
These models can be downloaded for local inference or used directly in the FastAPI app, and evaluated using ```eval.py```

## 🖥️ Running the FastAPI App
1. Install dependencies
```
pip install -r requirements.txt
```
2. Add a model
Option A — Local model
Place the model in ```app/models/<model_name>/```
Each model directory must contain:
* config.json
* pytorch_model.bin
* Tokenizer files
* label_map.json
* threshold.json (optional)
Option B — Download from Hugging Face
```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dpmendez/<model_name>")
model = AutoModelForSequenceClassification.from_pretrained("dpmendez/<model_name>")
```

3. Start the server
```
cd app
uvicorn main:app --reload
```

Then open:
``` http://127.0.0.1:8000 ```

### 🚧 Deployment Notes
Hosting multiple transformer models requires >512 MB RAM. Free-tier platforms may fail due to memory constraints. Local deployment is recommended for experimentation; production deployment would require model distillation or a GPU-enabled host.

## 📌 Limitations & Future Work
Limitations
* Binary labels oversimplify claim validity
* Dataset size limits generalization
* App predictions are indicative, not authoritative

Future Improvements
* Expand climate-domain datasets
* Multi-class or ordinal claim labels
* Model distillation for lighter deployment
* Visualization of confidence and uncertainty
