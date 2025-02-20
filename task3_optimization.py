import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from task1_implementation import NeuralNetwork

class OptimizedNeuralNetwork(NeuralNetwork):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, activation='sigmoid'):
        """
        Extended Neural Network with different activation functions
        Args:
            activation: Activation function to use ('sigmoid', 'tanh', or 'relu')
        """
        super().__init__(input_size, hidden_size, output_size, learning_rate)
        self.activation = activation
    
    def relu(self, Z):
        """ReLU activation function"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """Derivative of ReLU"""
        return (Z > 0).astype(float)
    
    def tanh(self, Z):
        """Tanh activation function"""
        return np.tanh(Z)
    
    def tanh_derivative(self, Z):
        """Derivative of tanh"""
        return 1 - np.tanh(Z)**2
    
    def forward_propagation(self, X):
        """Forward propagation with selected activation function"""
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        
        # Apply activation function
        if self.activation == 'relu':
            self.A1 = self.relu(self.Z1)
        elif self.activation == 'tanh':
            self.A1 = self.tanh(self.Z1)
        else:  # sigmoid
            self.A1 = self.sigmoid(self.Z1)
        
        # Output layer (always sigmoid for binary classification)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def backward_propagation(self, X, y):
        """Backward propagation with selected activation function"""
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dZ2_W2 = np.dot(dZ2, self.W2.T)
        
        # Apply activation derivative
        if self.activation == 'relu':
            dZ1 = dZ2_W2 * self.relu_derivative(self.Z1)
        elif self.activation == 'tanh':
            dZ1 = dZ2_W2 * self.tanh_derivative(self.Z1)
        else:  # sigmoid
            dZ1 = dZ2_W2 * self.sigmoid_derivative(self.A1)
        
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def experiment_hidden_neurons(X_train, X_test, y_train, y_test, neurons_list=[4, 8, 16, 32]):
    """Experiment with different numbers of hidden neurons"""
    results = []
    
    for neurons in neurons_list:
        model = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=neurons,
            output_size=1,
            learning_rate=0.01
        )
        
        model.train(X_train, y_train, epochs=500, batch_size=128)
        
        train_acc = model.get_accuracy(X_train, y_train)
        test_acc = model.get_accuracy(X_test, y_test)
        
        results.append({
            'neurons': neurons,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss_history': model.loss_history
        })
        
        print(f"\nHidden Neurons: {neurons}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
    
    return results

def experiment_activation_functions(X_train, X_test, y_train, y_test):
    """Experiment with different activation functions"""
    activations = ['sigmoid', 'tanh', 'relu']
    results = []
    
    for activation in activations:
        model = OptimizedNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=8,
            output_size=1,
            learning_rate=0.01,
            activation=activation
        )
        
        model.train(X_train, y_train, epochs=500, batch_size=128)
        
        train_acc = model.get_accuracy(X_train, y_train)
        test_acc = model.get_accuracy(X_test, y_test)
        
        results.append({
            'activation': activation,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss_history': model.loss_history
        })
        
        print(f"\nActivation Function: {activation}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
    
    return results

def experiment_learning_rates(X_train, X_test, y_train, y_test, lr_list=[0.001, 0.01, 0.1]):
    """Experiment with different learning rates"""
    results = []
    
    for lr in lr_list:
        model = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=8,
            output_size=1,
            learning_rate=lr
        )
        
        model.train(X_train, y_train, epochs=500, batch_size=128)
        
        train_acc = model.get_accuracy(X_train, y_train)
        test_acc = model.get_accuracy(X_test, y_test)
        
        results.append({
            'learning_rate': lr,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss_history': model.loss_history
        })
        
        print(f"\nLearning Rate: {lr}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
    
    return results

def plot_experiment_results(results, experiment_type):
    """Plot results from experiments"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss histories
    plt.subplot(1, 2, 1)
    for result in results:
        label = None
        if experiment_type == 'neurons':
            label = f'{result["neurons"]} neurons'
        elif experiment_type == 'activation':
            label = result['activation']
        elif experiment_type == 'learning_rate':
            label = f'lr={result["learning_rate"]}'
        
        plt.plot(result['loss_history'], label=label)
    
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    x = range(len(results))
    train_acc = [r['train_acc'] for r in results]
    test_acc = [r['test_acc'] for r in results]
    
    if experiment_type == 'neurons':
        labels = [str(r['neurons']) for r in results]
        plt.xlabel('Number of Hidden Neurons')
    elif experiment_type == 'activation':
        labels = [r['activation'] for r in results]
        plt.xlabel('Activation Function')
    else:
        labels = [str(r['learning_rate']) for r in results]
        plt.xlabel('Learning Rate')
    
    width = 0.35
    plt.bar([i - width/2 for i in x], train_acc, width, label='Train')
    plt.bar([i + width/2 for i in x], test_acc, width, label='Test')
    plt.xticks(x, labels)
    plt.ylabel('Accuracy')
    plt.title('Model Performance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
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
    
    print("\nOptimization Experiments")
    print("=" * 50)
    
    # 1. Experiment with different numbers of hidden neurons
    print("\n1. Hidden Neurons Experiment")
    neurons_results = experiment_hidden_neurons(X_train, X_test, y_train, y_test)
    plot_experiment_results(neurons_results, 'neurons')
    
    # 2. Experiment with different activation functions
    print("\n2. Activation Functions Experiment")
    activation_results = experiment_activation_functions(X_train, X_test, y_train, y_test)
    plot_experiment_results(activation_results, 'activation')
    
    # 3. Experiment with different learning rates
    print("\n3. Learning Rates Experiment")
    lr_results = experiment_learning_rates(X_train, X_test, y_train, y_test)
    plot_experiment_results(lr_results, 'learning_rate')

if __name__ == "__main__":
    main()