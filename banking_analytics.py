import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, classification_report, accuracy_score

# --- CONFIGURATION ---
# PLEASE UPDATE THIS PATH TO YOUR LOCAL DATASET LOCATION
FILE_PATH = 'bank-additional-full.csv' 

class BankingAnalytics:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.models = {}

    def load_and_preprocess(self):
        print(">> Loading Data...")
        # Load dataset with correct delimiter
        try:
            self.df = pd.read_csv(self.file_path, delimiter=';')
            print(f"Data Shape: {self.df.shape}")
        except FileNotFoundError:
            print("Error: Dataset file not found. Please check FILE_PATH.")
            return

        # 1. Handling Missing Values
        self.df.dropna(inplace=True)

        # 2. Feature Engineering (Task 3)
        self.df['total_contacts'] = self.df['campaign'] + self.df['previous']
        
        # 3. Label Encoding for 'education' (Ordinal)
        education_order = ['unknown', 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'professional.course', 'university.degree']
        self.df['education'] = pd.Categorical(self.df['education'], categories=education_order, ordered=True)
        self.df['education_encoded'] = self.df['education'].cat.codes

        # 4. Target Encoding (yes -> 1, no -> 0)
        self.df['y_encoded'] = self.df['y'].map({'no': 0, 'yes': 1})
        
        # 5. One-Hot Encoding for Nominal Columns (e.g. Job, Marital)
        # Note: We keep original df for exploration, creating a copy for training if needed
        print("Data Preprocessing Complete.\n")

    def run_regression_analysis(self):
        """ Task 2.1: Predict 'duration' using Age, Campaign, Previous """
        print("--- 1. Regression Analysis (Target: Duration) ---")
        X = self.df[['age', 'campaign', 'previous']]
        y = self.df['duration']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"R² Score: {r2_score(y_test, y_pred):.4f} (Note: Low correlation expected)")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}\n")

    def run_classification_analysis(self):
        """ Task 2.2 & 3: Predict Term Deposit (y) using Random Forest & Gradient Boosting """
        print("--- 2. Classification Analysis (Target: Subscription) ---")
        
        # Features selected based on report analysis
        features = ['age', 'duration', 'campaign', 'total_contacts']
        X = self.df[features]
        y = self.df['y_encoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # A. Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

        # B. Gradient Boosting (Final Selected Model)
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        
        print("\nGradient Boosting Report:")
        print(classification_report(y_test, y_pred_gb))
        
        # Feature Importance
        importances = gb_model.feature_importances_
        print("Feature Importance:")
        for name, imp in zip(features, importances):
            print(f"{name}: {imp:.4f}")
        print("")

    def run_clustering_analysis(self):
        """ Task 2.3: Customer Segmentation (K-Means) """
        print("--- 3. Clustering Analysis (Customer Segmentation) ---")
        X_cluster = self.df[['age', 'duration', 'campaign']]
        
        # K-Means with k=3 (as determined by Elbow Method in report)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(X_cluster)
        
        print("Clusters assigned. Cluster Centers:")
        print(kmeans.cluster_centers_)
        print("\nAnalysis Complete.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Initialize and Run Pipeline
    analytics = BankingAnalytics(FILE_PATH)
    analytics.load_and_preprocess()
    
    if analytics.df is not None:
        analytics.run_regression_analysis()
        analytics.run_classification_analysis()
        analytics.run_clustering_analysis()