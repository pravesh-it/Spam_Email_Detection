# Spam Email Detection Project

This project implements a spam email detection system using machine learning techniques.

## Project Goal

The goal of this project is to classify emails as either "spam" or "not spam" based on their content.

## Dataset

The project uses the `spam_email_dataset.csv` dataset, which contains email text and corresponding labels (spam or ham).

## Project Steps

1.  **Data Loading**: The dataset is loaded into a pandas DataFrame.
2.  **Data Exploration (EDA)**: Initial analysis is performed to understand the dataset, including checking for missing values and the distribution of spam and non-spam emails.
3.  **Text Preprocessing**: The email text undergoes cleaning and preparation, including:
    *   Converting text to lowercase
    *   Removing punctuation
    *   Tokenization
    *   Removing stop words
4.  **Text Vectorization**: The preprocessed text data is converted into numerical features using TF-IDF (Term Frequency-Inverse Data Frequency).
5.  **Data Splitting**: The dataset is split into training and testing sets (80% training, 20% testing) to evaluate the model's performance.
6.  **Model Training**: A Multinomial Naive Bayes model is trained on the training data.
7.  **Model Evaluation**: The trained model is evaluated using metrics such as accuracy, precision, recall, and F1-score on the test set.
8.  **Visualization**: The model's performance is visualized using a confusion matrix and a bar plot of the evaluation metrics.

## Key Findings

*   The dataset has an imbalanced distribution with more 'ham' emails than 'spam' emails.
*   The trained Multinomial Naive Bayes model achieved:
    *   Accuracy: 0.9188
    *   Precision: 1.0000
    *   Recall: 0.7133
    *   F1-score: 0.8327
*   The model demonstrates high precision for detecting spam, meaning that when it predicts an email is spam, it is very likely to be correct. However, the recall is lower, indicating that some spam emails are not being caught.

## Next Steps

*   Explore different text vectorization techniques (e.g., n-grams) to potentially improve performance.
*   Experiment with other classification algorithms (e.g., Support Vector Machines, Logistic Regression, or deep learning models).
*   Address the class imbalance issue using techniques like oversampling or undersampling to potentially improve recall.

## How to Run the Code

1.  Ensure you have the necessary libraries installed (`pandas`, `nltk`, `re`, `sklearn`, `matplotlib`, `seaborn`).
2.  Download the `spam_email_dataset.csv` file.
3.  Run the Python notebook cells sequentially.
