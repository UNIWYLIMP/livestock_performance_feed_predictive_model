# train_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import OneHotEncoder


def train_and_save_model(data_path, target_column, model_path):
    # Load the dataset from a CSV file
    df = pd.read_csv(data_path)

    # Separate the features (X) and the target (y)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_rep)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Save the trained model to a file
    joblib.dump(rf_classifier, model_path)
    print(f"Model saved to {model_path}")


# Example usage
if __name__ == "__main__":
    train_and_save_model('chickweight.csv', 'weight', 'feed_analyses_model.joblib')
