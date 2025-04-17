# Quora Insincere Questions Classification

This repository contains multiple machine learning models for classifying insincere questions on Quora. The models are implemented using different approaches to solve the binary classification problem of identifying insincere questions.

## Data Requirements

Before running any of the models, you need to download the training and testing data from the Kaggle competition:
1. Visit https://www.kaggle.com/competitions/quora-insincere-questions-classification/data
2. Download the following files:
   - `train.csv` - Contains the training data with question text and labels
   - `test.csv` - Contains the test data with question text
3. Place these files in the root directory of this project

## Implemented Models

### 1. Logistic Regression
A traditional machine learning approach using TF-IDF vectorization and logistic regression.
- Uses TF-IDF vectorization with 10,000 features
- Implements logistic regression with 500 maximum iterations
- Achieves validation accuracy of ~95.37%
- Good baseline model with relatively fast training time

### 2. XGBoost
A gradient boosting model that combines multiple weak learners to create a strong classifier.
- Uses TF-IDF vectorization for text preprocessing
- Implements XGBoost with binary logistic objective
- Achieves validation accuracy of ~83.09%
- Handles class imbalance through built-in mechanisms

### 3. Neural Network (CNN)
A deep learning approach using a Convolutional Neural Network architecture.
- Uses TF-IDF vectorization with 10,000 features
- Implements a sequential model with:
  - Input layer (10000 features)
  - Dense layer (256 units, ReLU activation)
  - Dropout (0.3)
  - Dense layer (128 units, ReLU activation)
  - Dropout (0.3)
  - Output layer (1 unit, sigmoid activation)
- Achieves validation accuracy of ~93.47%
- Includes early stopping and learning rate reduction callbacks

### 4. BERT
A state-of-the-art transformer-based model for natural language processing.
- Uses pre-trained BERT model for text encoding
- Implements a custom architecture with:
  - BERT preprocessor and encoder
  - Dropout layer (0.5)
  - Dense layer (16 units, ReLU activation)
  - Output layer (1 unit, sigmoid activation)
- Achieves competitive performance with transformer-based architecture
- Includes early stopping and learning rate reduction callbacks

## Usage

Each model is implemented in its own Jupyter notebook:
- `logisticRegression/logisticRegression.ipynb`
- `XGBoost/XGBoost.ipynb`
- `NeuralNetwork/NeuralNetwork.ipynb`
- `BERT/BERT.ipynb`

To run a specific model:
1. Ensure you have the required data files (`train.csv` and `test.csv`)
2. Open the corresponding notebook
3. Run all cells to train the model and generate predictions
4. The model will output a `submission.csv` file with predictions for the test set

## Dependencies

The project requires the following Python packages:
- numpy
- pandas
- scikit-learn
- tensorflow
- tensorflow-text
- tensorflow-hub
- xgboost
- matplotlib
- seaborn

## Results

Each model's performance can be evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The notebooks include visualization of training metrics and model performance. 