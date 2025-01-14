# NLP exploration - Sentiment analysis on disaster prediction tweets

This project focuses on classifying tweets to determine whether they are announcing a real disaster or not. It employs various machine learning techniques, from traditional methods like Bag-of-Words and TF-IDF to advanced deep learning models such as RNNs (GRU and LSTM) and Transformers (RoBERTa and DistilBERT).

## Overview

The project is structured as follows:

1. **Data Loading and Preprocessing**: Loads the dataset, performs basic cleaning, and applies advanced text preprocessing using spaCy for tokenization, lemmatization, and removal of stop words, URLs, emails, and short words.
2. **Traditional Machine Learning Models**: Implements Bag-of-Words and TF-IDF feature extraction techniques, coupled with Logistic Regression and XGBoost classifiers.
3. **Recurrent Neural Networks (RNNs)**: Explores the use of GRU and LSTM networks, both with and without an attention mechanism. It leverages pre-trained GloVe embeddings to enhance the model's understanding of word semantics.
4. **Transformers**: Utilizes state-of-the-art Transformer models, specifically RoBERTa and DistilBERT, for fine-tuning and classification.
5. **Evaluation and Comparison**: Evaluates each model's performance using metrics such as accuracy, loss, and training time. Plots training and validation curves to compare the learning process of different models.
6. **Inference**: Uses the best-performing model (DistilBERT) to classify the test dataset and generates a CSV file with predictions.

## Dependencies

-   Python 3.x
-   PyTorch 2.0.1+cu118
-   torchvision 0.15.2+cu118
-   torchtext 0.15.2
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
-   XGBoost
-   spaCy
-   Transformers
-   tqdm

## Usage

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/disaster-tweet-classification.git
    cd disaster-tweet-classification
    ```
2. **Place the Data:**
    Place `train.csv` and `test.csv` in the root directory of the project.
3. **Run the Notebook**:
    -   Open the Jupyter Notebook file (`Disaster_Tweet_Classification.ipynb`) using Jupyter or Google Colab.
    -   Execute the cells in order to perform data preprocessing, model training, evaluation, and inference.

## Results

The project compares the performance of various models. The DistilBERT model achieved the highest accuracy on the validation set (83.76%) and was subsequently used to generate predictions for the test set.

## Key Findings

-   Advanced text preprocessing significantly improves the performance of traditional machine learning models.=
-   Transformer models, particularly DistilBERT, provide superior performance compared to both traditional and RNN-based approaches.

## Further Improvements

-   Experiment with different hyperparameters, architectures, and pre-trained models.
-   Incorporate more sophisticated data augmentation techniques.
-   Explore ensemble methods to combine the strengths of different models.
-   Perform error analysis to identify common misclassifications and refine the models accordingly.
