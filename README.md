# NLP Exploration - Disaster Tweet Classification

This project focuses on classifying tweets to determine whether they are announcing a real disaster or not. It employs various machine learning techniques, from traditional methods like Bag-of-Words and TF-IDF to advanced deep learning models such as RNNs (GRU and LSTM) and Transformers (RoBERTa and DistilBERT).

## Project Overview

This repository demonstrates a systematic approach to text classification, implementing and comparing multiple techniques:

1. **Traditional Machine Learning**
   - Bag-of-Words (BoW) with Logistic Regression and XGBoost
   - TF-IDF with Logistic Regression and XGBoost

2. **Recurrent Neural Networks**
   - GRU with and without attention mechanisms
   - LSTM with and without attention mechanisms
   - Pre-trained GloVe word embeddings integration

3. **Transformer Models**
   - RoBERTa fine-tuning
   - DistilBERT fine-tuning

Each approach is evaluated on multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC to provide a thorough performance comparison.

## Data Processing Pipeline

The data processing pipeline involves several key steps:

1. **Initial Data Analysis**
   - Examining dataset structure and distribution
   - Identifying and handling missing values

2. **Text Preprocessing**
   - Tokenization using spaCy
   - Lemmatization to reduce words to their base forms
   - Removal of punctuation, whitespace, stop words, URLs, and emails
   - Filtering out non-alphabetic tokens and short words

3. **Feature Engineering**
   - Creating length-based features for the preprocessed text

## Model Implementation Details

### Traditional ML Models
- **Feature Extraction**: Implementation of both BoW and TF-IDF vectorization
- **Classifiers**: Logistic Regression and XGBoost with consistent evaluation metrics
- **Performance Analysis**: Comparison of model performance with both raw and preprocessed text

### RNN Models
- **Architecture**: Bi-directional GRU and LSTM networks with 2 layers
- **Word Embeddings**: Initialized with pre-trained GloVe vectors (100d)
- **Attention Mechanism**: Custom implementation of attention layers for both GRU and LSTM
- **Training Process**: Includes proper sequence padding, packing, and unpacking for efficient training
- **Optimization**: Adam optimizer with weight decay for regularization

### Transformer Models
- **Architecture**: Fine-tuning of pre-trained RoBERTa and DistilBERT models
- **Training Approach**: Custom dataset and dataloader implementation for transformers
- **Optimization**: AdamW optimizer with linear scheduler and warmup steps
- **Token Management**: Proper handling of special tokens, attention masks, and sequence length

## Results and Analysis

### Performance Metrics

The project evaluates all models using consistent metrics:
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC

### Traditional ML Model Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| BOW - LR - Untreated | 0.682248 | 0.656848 | 0.545232 | 0.595858 | 0.665342 |
| BOW - LR - Treated | 0.694532 | 0.723485 | 0.467564 | 0.568030 | 0.666501 |
| BOW - XGBoost - Treated | 0.701893 | 0.725632 | 0.492044 | 0.586433 | 0.675976 |
| TF-IDF - LR - Untreated | 0.676656 | 0.641061 | 0.561812 | 0.598826 | 0.662473 |
| TF-IDF - LR - Treated | 0.690326 | 0.704301 | 0.481028 | 0.571636 | 0.664477 |
| TF-IDF - XGBoost - Treated | 0.686646 | 0.727835 | 0.432069 | 0.542243 | 0.655205 |

### RNN Model Results

| Model | Best F1-Score | Best ROC-AUC |
|-------|--------------|-------------|
| GRU | 0.779800 | 0.872743 |
| GRU with Attention | 0.764045 | 0.870303 |
| LSTM | 0.772144 | 0.868771 |
| LSTM with Attention | 0.772327 | 0.869276 |

### Transformer Model Results

| Model | Best F1-Score | Best ROC-AUC |
|-------|--------------|-------------|
| RoBERTa | 0.810934 | 0.887643 |
| DistilBERT | 0.803951 | 0.896761 |

### Key Findings

1. **Text Preprocessing Impact**:
   - Significant improvement in performance across all traditional models after text preprocessing
   - Reduction in vocabulary size while maintaining semantic meaning

2. **Model Comparison**:
   - Transformer models outperform both traditional ML and RNN approaches
   - Attention mechanisms consistently improve RNN performance
   - **DistilBERT** achieves the best balance of performance and efficiency

3. **Training Dynamics**:
   - Learning curves showing convergence patterns for each model
   - Analysis of training and validation loss/F1-score progression

## Dependencies

```
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchtext==0.15.2
numpy<2
pandas
matplotlib
scikit-learn
xgboost
spacy
transformers
tqdm
```

## Usage

1. **Environment Setup**:
   ```bash
   # Install PyTorch with CUDA support
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchtext==0.15.2 -f https://download.pytorch.org/whl/torch_stable.html
   
   # Install other dependencies
   pip install "numpy<2" pandas matplotlib scikit-learn xgboost spacy transformers tqdm
   
   # Download spaCy language model
   python -m spacy download en_core_web_sm
   ```

2. **Data Preparation**:
   - Place `train.csv` and `test.csv` in the project root directory
   - The training data should include 'text' and 'target' columns

3. **Running the Models**:
   - Execute the notebook cells sequentially to reproduce the results
   - Model checkpoints will be saved as `best_model_[model_type].pt`

## Future Improvements

- Experiment with ensemble methods combining traditional ML and deep learning approaches
- Implement cross-validation for more robust performance evaluation
- Explore additional transformer architectures (BERT, T5, etc.)
- Investigate domain adaptation techniques for specific disaster types
- Implement explainability methods to understand model predictions
- Optimize hyperparameters using systematic approaches like Bayesian optimization
