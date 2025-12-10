## Climate Misinformation Detection 🌍

This project applies **Natural Language Processing (NLP)** and **transformer-based models (LLMs)** to detect misinformation in climate and environmental news. It builds on existing fake news datasets and extends toward applications in climate change, renewable energy, and conservation, aligning with real-world social-good impact.

A lightweight FastAPI web app is included so users can test multiple models interactively through a browser interface.

### 🔍 Problem Statement
Misinformation around climate change and conservation undermines public trust, policy adoption, and sustainable action. 

The goal of this project is to:
* Compare baseline ML models against modern LLM-based classifiers.
* Enable fact-checking tools for environmental communication.
* Support NGOs and policymakers in identifying climate misinformation.
* Provide a scalable pipeline to monitor public discourse around conservation issues.

#### 📊 Data

* Climate Fever fact-check articles (environment-specific ground truth), dowloaded from https://huggingface.co/datasets/tdiggelm/climate_fever
* Scraped climate news from Science Feedback, from https://science.feedback.org/climate-feedback/

Data is preprocessed into train/validation/test splits with balanced label distributions.

#### ⚙️ Methods

* Preprocessing: Tokenization, text cleaning, label encoding.
* Models:
  * Baseline: Logistic Regression, Random Forests, Linear SVC, XGBoost (TF-IDF).
  * Transformer models: distilbert, bert-base-uncased, electra-base-discriminator.
* Training:
  * Implemented in PyTorch + Hugging Face Transformers.
  * Optimizer: AdamW, with learning rate scheduling.
* Evaluation Metrics: Balanced accuracy, f1-score (weighted), precision and recall (false).

#### 🚀 Results (coming soon...)

| Model                      | Balanced Accuracy | F1 (weighted) | Precision (false) | Recall (false)|
| -------------------------- | ----------------- | --------------| ----------------- | ------------- |
| Baseline: TF-IDF + XGBoost | XX%               | XX            | XX                | XX            |
| DistilBERT Fine-tuned      | XX%               | XX            | XX                | XX            |
| BERT-base Fine-tuned       | XX%               | XX            | XX                | XX            |

**Observations:**

### 🖥️ Running the App (FastAPI + HTML)
The project includes a FastAPI web interface for testing models.
1. Install dependencies
```cpp pip install -r requirements.txt ```

3. Add or download a model
You can either:
Option A — Use your own fine-tuned model
Place it in:
```cpp app/models/<model_name>/ ```
Each model directory must contain:
```cpp
config.json
pytorch_model.bin
tokenizer files
label_map.json
threshold.json (optional)
```
Option B — Download from Hugging Face
```cpp
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("dpmendez/<modelname>")
model = AutoModelForSequenceClassification.from_pretrained("<dpmendez/modelname>")
```

4. Start the local server
From the project root:
```cpp
cd app
uvicorn main:app --reload
```
Then open:
```cpp
http://127.0.0.1:8000
```
