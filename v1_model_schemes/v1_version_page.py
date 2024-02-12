# 1. load the datasets - by config
#   1.1 split for train test (if validation set is needed it will this function again)
#   1.2 - split the data for each model
# 2. data pipelines - for each model - create data pipeline for each datafreame in a loop, then append the model
# 3. model as a transformer
#  4. aggregate the results to one df - (feature union) - example in run models.ipynb
# 5. model evaluation
# 6. model selection


# pipeline.fit_transform(X_train,y_train)
#pipeline.transform(X_test)


#### the implemetation shuold be close to this:
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define transformers
transformer_list = [
    ('scaler', StandardScaler()),  # Standardize features
    ('pca', PCA(n_components=10))  # Reduce dimensionality
]

# Define a FeatureUnion combining the transformers
feature_union = FeatureUnion(transformer_list)

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a pipeline
pipeline = Pipeline([
    ('features', feature_union),  # Apply the feature union
    ('classifier', model)  # Apply the classifier
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
