## Climate Misinformation Detection 🌍

This project applies **Natural Language Processing (NLP)** and **transformer-based models (LLMs)** to detect misinformation in climate and environmental news. It builds on existing fake news datasets and extends toward applications in climate change, renewable energy, and conservation, aligning with real-world social-good impact.

### 🔍 Problem Statement

Misinformation around climate change and conservation undermines public trust, policy adoption, and sustainable action.
The goal of this project is to:
* Detect false or misleading environmental news articles.
* Benchmark NLP/LLM models for misinformation detection.
* Explore extensions into fact-checking, explainability, and cross-domain evaluation (general fake news → climate-specific).

#### 📊 Data

* Climate Feedback fact-check articles (environment-specific ground truth).
* Scraped climate news from Science Feedback.

Data is preprocessed into train/validation/test splits with balanced label distributions.

#### ⚙️ Methods

* Preprocessing: Tokenization, text cleaning, label encoding.
* Models:
  * Baseline: Logistic Regression, XGBoost (bag-of-words, TF-IDF).
  * Transformer models: bert-base-uncased, roberta-base, distilbert.
* Training:
  * Implemented in PyTorch + Hugging Face Transformers.
  * Optimizer: AdamW, with learning rate scheduling.
* Evaluation Metrics: Accuracy, F1-score, Precision/Recall.

#### 🚀 Results (to update after experiments)

* Baseline (TF-IDF + XGBoost): ~XX% accuracy on general dataset.
* RoBERTa fine-tuned: ~YY% accuracy, improved F1 on environmental subset.
* Observations: ...

### 🌱 Social Good Impact

This project aims to demonstrates how NLP and LLMs can be leveraged to:
* Support NGOs and policymakers in identifying climate misinformation.
* Enable fact-checking tools for environmental communication.
* Provide a scalable pipeline to monitor public discourse around conservation issues.
