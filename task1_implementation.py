import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize neural network with specified architecture
        Args:
            input_size: Number of features (13 for heart disease dataset)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons (1 for binary classification)
            learning_rate: Learning rate for gradient descent (default 0.01)
        """
        # Initialize weights using uniform distribution between -1 and 1
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        
        # Initialize biases to 0
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        
        # Lists to store metrics
        self.loss_history = []
        self.accuracy_history = []
        
    def sigmoid(self, Z):
        """
        Sigmoid activation function: f(z) = 1 / (1 + e^(-z))
        """
        return 1 / (1 + np.exp(-Z))
    
    def sigmoid_derivative(self, A):
        """
        Derivative of sigmoid: f'(z) = f(z) * (1 - f(z))
        """
        return A * (1 - A)
    
    def forward_propagation(self, X):
        """
        Forward propagation implements:
        Z1 = X·W1 + b1
        A1 = sigmoid(Z1)
        Z2 = A1·W2 + b2
        A2 = sigmoid(Z2)
        """
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear transformation
        self.A1 = self.sigmoid(self.Z1)         # Activation
        
        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def compute_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss:
        L = -1/m * Σ(y·log(ŷ) + (1-y)·log(1-ŷ))
        """
        m = y_true.shape[0]
        loss = -(1/m) * np.sum(y_true * np.log(y_pred + 1e-15) + 
                              (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss
    
    def backward_propagation(self, X, y):
        """
        Backward propagation implements:
        dZ2 = A2 - y
        dW2 = (1/m) * A1ᵀ·dZ2
        db2 = (1/m) * Σ(dZ2)
        dZ1 = dZ2·W2ᵀ ⊙ sigmoid_derivative(A1)
        dW1 = (1/m) * Xᵀ·dZ1
        db1 = (1/m) * Σ(dZ1)
        """
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    def update_parameters(self, gradients):
        """
        Update parameters using gradient descent:
        W = W - α·dW
        b = b - α·db
        """
        self.W1 -= self.learning_rate * gradients["dW1"]
        self.b1 -= self.learning_rate * gradients["db1"]
        self.W2 -= self.learning_rate * gradients["dW2"]
        self.b2 -= self.learning_rate * gradients["db2"]
    
    def get_accuracy(self, X, y):
        """Calculate classification accuracy"""
        predictions = self.forward_propagation(X)
        predictions = (predictions >= 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def create_mini_batches(self, X, y, batch_size):
        """Create mini-batches for training"""
        m = X.shape[0]
        mini_batches = []
        
        # Shuffle data
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        
        # Create mini-batches
        num_complete_batches = m // batch_size
        for k in range(num_complete_batches):
            X_mini = X_shuffled[k * batch_size:(k + 1) * batch_size]
            y_mini = y_shuffled[k * batch_size:(k + 1) * batch_size]
            mini_batches.append((X_mini, y_mini))
        
        # Handle the end case (last mini-batch < batch_size)
        if m % batch_size != 0:
            X_mini = X_shuffled[num_complete_batches * batch_size:]
            y_mini = y_shuffled[num_complete_batches * batch_size:]
            mini_batches.append((X_mini, y_mini))
        
        return mini_batches
    
    def train(self, X, y, epochs, batch_size):
        """Train the neural network using mini-batch gradient descent"""
        for epoch in range(epochs):
            mini_batches = self.create_mini_batches(X, y, batch_size)
            epoch_loss = 0
            
            for X_mini, y_mini in mini_batches:
                # Forward propagation
                y_pred = self.forward_propagation(X_mini)
                
                # Compute loss
                batch_loss = self.compute_loss(y_mini, y_pred)
                epoch_loss += batch_loss
                
                # Backward propagation
                gradients = self.backward_propagation(X_mini, y_mini)
                
                # Update parameters
                self.update_parameters(gradients)
            
            # Calculate epoch metrics
            epoch_loss /= len(mini_batches)
            epoch_accuracy = self.get_accuracy(X, y)
            
            # Store metrics
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)
            
            # Print progress every 50 epochs
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Loss: {epoch_loss:.4f}")
                print(f"Accuracy: {epoch_accuracy:.4f}\n")

def plot_training_metrics(model):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(model.accuracy_history)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
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
    
    # Create and train model with specified parameters
    model = NeuralNetwork(
        input_size=13,  # 13 features
        hidden_size=8,  # Can be modified in Task 3
        output_size=1,  # Binary classification
        learning_rate=0.01  # As specified
    )
    
    # Train with specified parameters
    model.train(
        X=X_train,
        y=y_train,
        epochs=500,  # As specified
        batch_size=128  # As specified
    )
    
    # Plot training metrics
    plot_training_metrics(model)
    
    # Print final test accuracy
    test_accuracy = model.get_accuracy(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")