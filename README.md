# Predictive Analytics for Banking Customer Behavior

## Project Overview
This project leverages machine learning to analyze customer behavior and optimize marketing strategies for a banking institution. Using the **UCI Bank Marketing Dataset** (41,188 records), the goal is to predict term deposit subscriptions and reduce customer churn.

The project demonstrates an end-to-end data science lifecycle, from data preprocessing (Label/One-Hot Encoding) to building advanced predictive models.

## Key Techniques & Models
* **Classification:** Built **Random Forest** and **Gradient Boosting** models to predict customer response (`yes`/`no`).
* **Clustering:** Applied **K-Means Clustering (K=3)** to segment customers based on age, campaign frequency, and call duration.
* **Regression:** Analyzed the relationship between customer demographics and call duration using Linear Regression.
* **Feature Engineering:** Created new features like `total_contacts` to improve model performance.

## Key Results & Insights
1.  **Top Predictor:** Call duration is the most critical factor (79.5% importance) in predicting subscription success.
2.  **Model Performance:** The Gradient Boosting classifier achieved **90% Accuracy** and **60% Precision** for positive predictions.
3.  **Customer Segments:** Identified 3 distinct customer groups (e.g., Young/Low Balance vs. Middle-Aged/High Balance) to enable targeted marketing.

## Tech Stack
* **Python:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForest, GradientBoosting, KMeans, LinearRegression)
* **Visualization:** Matplotlib
* **Preprocessing:** StandardScaler, LabelEncoder

## How to Run
1.  Clone the repository.
2.  Ensure you have the `bank-additional-full.csv` dataset.
3.  Run `banking_analytics.py` to execute the preprocessing, training, and evaluation pipeline.
