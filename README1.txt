Credit Card Fraud Detection
Project Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used in this project is highly imbalanced, with a very small percentage of transactions being fraudulent. The goal is to build a model that can accurately identify fraudulent transactions while minimizing false positives.

Dataset
The dataset used in this project is sourced from Kaggle and contains transactions made by European cardholders in September 2013. It has 284,807 transactions, out of which 492 are fraudulent (approximately 0.17%).

Features: The dataset contains 30 features, most of which have been anonymized (V1, V2, ..., V28) due to confidentiality issues. The only features that have not been anonymized are Time, Amount, and Class (which indicates whether a transaction is fraudulent).
Target: The Class column indicates the label, where 1 represents a fraudulent transaction and 0 represents a legitimate one.
Project Structure
credit_card_fraud_detection.ipynb: The Jupyter Notebook containing the code, analysis, and results of the project.
README1.txt: This file, providing an overview of the project, instructions for setting up the environment, and details on how to run the project.
Machine Learning Techniques Used
Data Preprocessing:

Handling missing values (if any).
Normalization of numerical features like Amount and Time.
Splitting the dataset into training and testing sets.
Exploratory Data Analysis (EDA):

Understanding the distribution of the features.
Visualizing the correlation between different features.
Addressing data imbalance using techniques like SMOTE (Synthetic Minority Over-sampling Technique).
Modeling:

Logistic Regression
Decision Trees
Random Forest
Gradient Boosting Machines (XGBoost, LightGBM)
Neural Networks
Model Evaluation:

Confusion Matrix
Precision, Recall, F1-Score
ROC-AUC Curve
Setup and Installation
Prerequisites
Python 3.7 or higher
Jupyter Notebook
Required libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, imbalanced-learn, XGBoost, LightGBM
Installation
Clone the repository:


git clone https://github.com/iamsanusinghh/credit-card-fraud-detection.git
cd credit-card-fraud-detection
Install the required libraries:



pip install -r requirements.txt
Run the Jupyter Notebook:



jupyter notebook credit-card-fraud-detection.ipynb
Results
The best performing model achieved a high precision and recall for the minority class (fraudulent transactions). The results are discussed in detail in the Jupyter Notebook, along with visualizations to aid in understanding the performance of each model.

Conclusion
The project demonstrates the effectiveness of various machine learning techniques in detecting fraudulent transactions. While the dataset is highly imbalanced, the models have been tuned to minimize false positives and maximize the detection of actual fraudulent transactions.

Future Work
Improving Model Performance: Experimenting with more advanced techniques like ensemble learning or deep learning.
Real-Time Detection: Implementing the model in a real-time transaction processing system.
Feature Engineering: Exploring the creation of new features that could enhance model performance.