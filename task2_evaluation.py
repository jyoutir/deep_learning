import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from task1_implementation import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ModelEvaluator:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        """
        Initialize evaluator with model and data
        Args:
            model: Trained NeuralNetwork instance
            X_train, X_test: Training and test features
            y_train, y_test: Training and test labels
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def plot_learning_curves(self):
        """Plot training loss and accuracy curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss curve
        plt.subplot(1, 2, 1)
        plt.plot(self.model.loss_history, label='Training Loss')
        plt.title('Learning Curve: Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross-Entropy Loss')
        plt.legend()
        
        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(self.model.accuracy_history, label='Training Accuracy')
        plt.title('Learning Curve: Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for test set predictions"""
        # Get predictions
        y_pred = self.model.forward_propagation(self.X_test)
        y_pred_classes = (y_pred >= 0.5).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred_classes)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_classes))
    
    def plot_roc_curve(self):
        """Plot ROC curve and calculate AUC"""
        # Get predictions
        y_pred_proba = self.model.forward_propagation(self.X_test)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def evaluate_model_stability(self, n_runs=5):
        """
        Evaluate model stability across multiple runs
        Args:
            n_runs: Number of training runs to perform
        """
        test_accuracies = []
        train_accuracies = []
        
        print("\nModel Stability Analysis:")
        print("-" * 50)
        
        for i in range(n_runs):
            # Create and train new model instance
            model = NeuralNetwork(
                input_size=self.X_train.shape[1],
                hidden_size=8,
                output_size=1,
                learning_rate=0.01
            )
            
            # Train model
            model.train(
                X=self.X_train,
                y=self.y_train,
                epochs=500,
                batch_size=128
            )
            
            # Get accuracies
            train_acc = model.get_accuracy(self.X_train, self.y_train)
            test_acc = model.get_accuracy(self.X_test, self.y_test)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"Run {i+1}:")
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}\n")
        
        # Print summary statistics
        print("Summary Statistics:")
        print(f"Mean Training Accuracy: {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}")
        print(f"Mean Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")

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
    
    # Create and train model
    model = NeuralNetwork(
        input_size=13,
        hidden_size=8,
        output_size=1,
        learning_rate=0.01
    )
    
    # Train model
    model.train(
        X=X_train,
        y=y_train,
        epochs=500,
        batch_size=128
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(model, X_train, X_test, y_train, y_test)
    
    # Perform comprehensive evaluation
    print("\nComprehensive Model Evaluation")
    print("=" * 50)
    
    # 1. Plot learning curves
    print("\n1. Learning Curves Analysis")
    evaluator.plot_learning_curves()
    
    # 2. Confusion matrix and classification metrics
    print("\n2. Classification Performance Analysis")
    evaluator.plot_confusion_matrix()
    
    # 3. ROC curve analysis
    print("\n3. ROC Curve Analysis")
    evaluator.plot_roc_curve()
    
    # 4. Model stability analysis
    print("\n4. Model Stability Analysis")
    evaluator.evaluate_model_stability()

if __name__ == "__main__":
    main()