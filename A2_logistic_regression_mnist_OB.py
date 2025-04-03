"""
@author:  Bracci Lorenzo - Federica Musazzi - Schiavi Francesco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math

class LogisticRegressionMNIST:
    def __init__(self):
        """
        Initialize all configuration parameters and data structures.
        """
        # Data configuration
        self.data_path = 'mnist.csv'  # Path to MNIST dataset
        self.n_train = 100           # Number of training samples
        self.lambda_0 = 1            # Regularization parameter
        
        # Training parameters
        self.num_steps = 1000        # Max iterations for Newton-Raphson
        self.tolerance = 1e-5         # Convergence tolerance
        
        # Visualization parameters
        self.offset = 200            # Offset for probability bar plots
        self.n_images = 20           # Number of images to visualize
        self.images_per_row = 10      # Images per row in plots
        
        # Data storage attributes
        self.df = None               # Raw dataframe
        self.x_train = None          # Training features
        self.y_train = None          # Training labels
        self.x_test = None           # Test features
        self.y_test = None           # Test labels
        self.beta = None             # Model coefficients
        
    def load_data(self):
        """Load and preprocess MNIST data, filter 0/1 digits"""
        # Read CSV and convert to numpy arrays
        self.df = pd.read_csv(self.data_path)
        y_labels_data = self.df['label'].to_numpy()
        x_features_data = self.df.drop(columns='label').to_numpy()
        
        # Filter only 0/1 digits
        idx = np.where(y_labels_data <= 1)[0]
        self.y_train = y_labels_data[idx][:self.n_train]
        self.x_train = x_features_data[idx][:self.n_train]
        
        # Create test set from remaining samples
        n_total = len(idx)
        self.y_test = y_labels_data[idx][self.n_train:n_total]
        self.x_test = x_features_data[idx][self.n_train:n_total]
        
    def logistic(self, x):
        """Logistic sigmoid function"""
        return 1 / (1 + np.exp(-x))
    
    def logistic_forecast(self, features, beta):
        """Generate class predictions using model coefficients"""
        signal_hat = np.dot(features, beta)
        y_hat = np.sign(signal_hat)
        y_hat[y_hat < 0] = 0
        return y_hat
    
    def prediction_accuracy(self, y_pred, y_true):
        """Calculate classification accuracy"""
        errors = np.abs(y_pred - y_true)
        return 1 - sum(errors)/len(y_pred)
    
    def train_model(self):
        """Train regularized logistic regression using Newton-Raphson"""
        # Initialize coefficients
        self.beta = np.zeros(self.x_train.shape[1])
        
        for step in range(self.num_steps):
            p = self.logistic(self.x_train @ self.beta)
            W = np.diag(p * (1 - p))
            
            # Compute gradient with regularization
            gradient = -self.x_train.T @ (self.y_train - p) + 2*self.lambda_0*self.beta
            
            if np.linalg.norm(gradient) > self.tolerance:
                # Compute regularized Hessian
                Hessian = self.x_train.T @ W @ self.x_train + 2*self.lambda_0*np.eye(self.x_train.shape[1])
                # Update coefficients
                self.beta -= np.linalg.solve(Hessian, gradient)
            else:
                break
                
    def evaluate(self):
        """Evaluate model on test set and print metrics"""
        y_hat = self.logistic_forecast(self.x_test, self.beta)
        acc = self.prediction_accuracy(y_hat, self.y_test)
        print(f"\nPrediction Accuracy: {acc*100:.2f}%")
        return y_hat
    
    def plot_digits(self):
        """Plot sample training digits"""
        plt.figure(figsize=(25,5))
        for idx, (image, label) in enumerate(zip(self.x_train[5:10], self.y_train[5:10])):
            plt.subplot(1, 5, idx+1)
            plt.imshow(image.reshape(28,28), cmap=plt.cm.gray)
            plt.title(f'Label: {label}\n', fontsize=20)
        plt.savefig('training_digits.png')
        plt.close()
        
    def plot_confusion_matrix(self, y_hat):
        """Generate confusion matrix visualization"""
        cm = confusion_matrix(self.y_test, y_hat)
        disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix")
        plt.savefig('confusion_matrix.png')
        plt.close()
        
    def _draw_bars(self, ax, prob, pred, true_label):
        """Helper function for probability bar plots"""
        bars = ax.bar(range(2), [1-prob, prob])
        ax.set_ylim([0,1])
        ax.set_xticks(range(2))
        
        color = "green" if pred == true_label else "red"
        bars[int(pred)].set_color(color)
        
    def plot_probability_bars(self, y_hat):
        """Visualize class probabilities for test samples"""
        probs = self.logistic(np.dot(self.x_test, self.beta))
        
        n_rows = 2 * math.ceil(self.n_images / self.images_per_row)
        fig, axs = plt.subplots(n_rows, self.images_per_row, 
                              figsize=(3*self.images_per_row, 3*n_rows))
        
        row = col = 0
        for i in range(self.n_images):
            # Plot image
            axs[2*row, col].imshow(self.x_test[self.offset+i].reshape(28,28), cmap="gray")
            axs[2*row, col].set_title(int(self.y_test[self.offset+i]))
            axs[2*row, col].axis("off")
            
            # Plot probability bars
            self._draw_bars(axs[2*row+1, col], 
                          probs[self.offset+i], 
                          y_hat[self.offset+i],
                          self.y_test[self.offset+i])
            
            col += 1
            if col == self.images_per_row:
                col = 0
                row += 1
        fig.savefig('probability_bars.png')
        plt.close(fig)

if __name__ == "__main__":
    # Part (c)
    print("\n" + "="*50 + "\nPart (c): Matrix Singularity Analysis\n" + "="*50)
    model_c = LogisticRegressionMNIST()
    model_c.lambda_0 = 0  # Disable regularization
    model_c.load_data()
    
    # Compute rank of feature matrix
    rank = np.linalg.matrix_rank(model_c.x_train)
    print(f"\nRank of x_train: {rank} (out of {model_c.x_train.shape[1]} features)")
    print("This indicates rank deficiency - columns are linearly dependent.")
    
    # Attempt training (will fail)
    try:
        model_c.train_model()
    except np.linalg.LinAlgError as e:
        print(f"\nTraining failed: {str(e)}")
        print("Singular matrix encountered in Newton-Raphson update.")

    # Part (d): Run with regularization to fix singularity
    print("\n\n" + "="*50 + "\nPart (d): Regularized Solution\n" + "="*50)
    model_d = LogisticRegressionMNIST()
    model_d.lambda_0 = 1  # Default regularization
    model_d.load_data()
    
    # Visualization of training digits
    model_d.plot_digits()
    
    # Model training with regularization
    print("\nTraining regularized model...")
    model_d.train_model()
    print("Model training completed.")
    
    # Evaluation and visualization
    predictions = model_d.evaluate()
    model_d.plot_confusion_matrix(predictions)
    model_d.plot_probability_bars(predictions)