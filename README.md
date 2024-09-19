import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class ColorPrediction:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = None
    
    def preprocess_data(self):
        # Fill missing values
        self.data.fillna(method='ffill', inplace=True)
        
        # Convert categorical columns to numerical
        for col in self.data.select_dtypes(include=['object']).columns:
            self.data[col] = self.data[col].astype('category').cat.codes

        # Separate features and target variable
        self.X = self.data.drop('target', axis=1)  # Features
        self.y = self.data['target']                # Target variable
        
        # Scale features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)
    
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
