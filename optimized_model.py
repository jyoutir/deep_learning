from task3_optimization import OptimizedNeuralNetwork
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = pd.read_csv('heart-disease.csv')
X = data.drop('target', axis=1).values
y = data['target'].values.reshape(-1, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create optimized model with best parameters
optimized_model = OptimizedNeuralNetwork(
    input_size=13,
    hidden_size=16,  # Best performance with 16 neurons
    output_size=1,
    learning_rate=0.1,  # Best performance with 0.1
    activation='relu'  # Best performance with ReLU
)

# Train model
optimized_model.train(
    X=X_train,
    y=y_train,
    epochs=500,
    batch_size=128
)

# Evaluate final performance
train_accuracy = optimized_model.get_accuracy(X_train, y_train)
test_accuracy = optimized_model.get_accuracy(X_test, y_test)

print("\nOptimized Model Performance")
print("=" * 50)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")