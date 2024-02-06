import sys
import joblib

def load_model(path):
    # Load the trained pipeline
    return joblib.load(path)

def predict(model, input_data):
    # Predict with the pipeline model
    return model.predict([input_data])

if __name__ == "__main__":
    # Load the model
    model = load_model('model.joblib')
    
    # Convert command line arguments to suitable input format
    # This example assumes numerical input
    input_data = list(map(float, sys.argv[1:]))
    
    # Make predictions
    predictions = predict(model, input_data)
    
    # Print predictions
    print("Predictions:", predictions)