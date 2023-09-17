
# BUILDING A MODEL THAT CAN RATE THE SENTIMENT OF A TWEET BASED ON ITS CONTENT


## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Conclusions](#conclusions)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## About

This project focuses on building a Natural Language Processing (NLP) model to analyze Twitter sentiment regarding Apple and Google products. The dataset used in this project contains over 9,000 tweets that have been manually rated as positive, negative, or neutral by human raters.

The primary goal of this project is to develop a model that can classify tweets into positive and negative sentiments accurately. Various NLP techniques and machine learning models have been employed and evaluated to achieve this objective.

## Dataset

- The dataset used for this project was obtained from CrowdFlower via data.world.
- It contains over 9,000 tweets related to Apple and Google products, along with corresponding sentiment labels.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Jupyter Notebook


## Usage

- The Jupyter Notebook `sentiment_analysis.ipynb` provides a step-by-step guide to the sentiment analysis process.
- You can modify and extend the analysis to suit your specific requirements or use case.

## Models

Three machine learning models were employed and evaluated for sentiment analysis:

1. **Logistic Regression Model**
   - Achieved an accuracy of 85%.
   - Performs better in identifying positive sentiment than negative sentiment.

2. **Multinomial Naive Bayes Model**
   - Achieved an accuracy of 85%.
   - Excellent precision in identifying positive sentiment but poor recall for negative sentiment.

3. **Support Vector Machine (SVM) Model**
   - Achieved the highest accuracy of 89%.
   - Balanced precision and recall scores for both positive and negative sentiment.

## Results

The results of the sentiment analysis models are summarized below:

- Logistic Regression Model:
  - Accuracy: 85%
  - F1-score (Positive): 0.92
  - F1-score (Negative): 0.12

- Multinomial Naive Bayes Model:
  - Accuracy: 85%
  - F1-score (Positive): 0.92
  - F1-score (Negative): 0.05

- Support Vector Machine (SVM) Model:
  - Accuracy: 89%
  - F1-score (Positive): 0.94
  - F1-score (Negative): 0.47

## Conclusions

- The Support Vector Machine (SVM) model outperformed the other models with the highest accuracy and balanced performance in identifying both positive and negative sentiment.
- Addressing class imbalance and fine-tuning the SVM model are recommended for further improvements.
- Regular model updates are essential for maintaining effectiveness in real-time sentiment analysis.


