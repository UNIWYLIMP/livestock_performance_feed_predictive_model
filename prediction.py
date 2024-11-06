# predict.py

import pandas as pd
import joblib

def load_model_and_predict(model_path, new_data):
    # Load the trained model
    model = joblib.load(model_path)
    
    # If new_data is a DataFrame, convert it to a numpy array
    if isinstance(new_data, pd.DataFrame):
        new_data = new_data.values
    
    # Make predictions
    predictions = model.predict(new_data)
    
    return predictions

# Example usage
if __name__ == "__main__":
    # Example new data for prediction
    # Replace this with your actual real-time data
    example_data = pd.DataFrame([{
        'Time': 15,
        'Chick': 1,
        'Diet': 1
    }])
    
    predictions = load_model_and_predict('feed_analyses_model.joblib', example_data)
    print("New Predictions:")
    print(predictions)
