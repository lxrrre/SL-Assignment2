import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

class DataLoader:
    """Loads and preprocesses CIFAR-10 data batches."""
    def __init__(self, data_paths, test_path, meta_path):
        self.data_paths = data_paths
        self.test_path = test_path
        self.meta_path = meta_path

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    def load_data(self):
        # CIFAR-10 images are 32x32 RGB, flattened to 3072 features (32*32*3)
        train_batches = [self.unpickle(path) for path in self.data_paths]
        X_train = np.concatenate([batch['data'] for batch in train_batches])
        y_train = np.concatenate([batch['labels'] for batch in train_batches])
        
        test_batch = self.unpickle(self.test_path)
        X_test, y_test = test_batch['data'], test_batch['labels']
        
        meta_data = self.unpickle(self.meta_path)
        label_names = meta_data['label_names']
        
        return X_train, y_train, X_test, y_test, label_names

class Preprocessor:
    """Preprocesses image data by flattening and standardizing features."""
    def __init__(self):
        self.scaler = StandardScaler()

    def flatten(self, X):
        return X.reshape(-1, 32*32*3)

    def scale(self, X_train, X_test):
        # StandardScaler ensures zero mean and unit variance for stable optimization
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

class Visualizer:
    """Visualizes images and cross-validation metrics."""
    @staticmethod
    def _reshape_image(image_array):
        return image_array.reshape(3, 32, 32).transpose(1, 2, 0)

    def plot_image(self, image_array, title=""):
        plt.imshow(self._reshape_image(image_array))
        plt.title(title)
        plt.axis('off')
        plt.show()

    def plot_scores(self, C_values, scores, score_name, color):
        plt.figure(figsize=(8, 5))
        plt.plot(C_values, scores, marker='o', linestyle='-', color=color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('C (Regularization Strength)')
        plt.ylabel(score_name)
        plt.title(f'{score_name} vs. Regularization Strength')
        plt.grid(True, which='both', linestyle='--')
        plt.show()

def main():
    DATA_PATHS = [
        "./cifar-10-batches-py/data_batch_1",
        "./cifar-10-batches-py/data_batch_2",
        "./cifar-10-batches-py/data_batch_3",
        "./cifar-10-batches-py/data_batch_4",
        "./cifar-10-batches-py/data_batch_5"
    ]
    TEST_PATH = "./cifar-10-batches-py/test_batch"
    META_PATH = "./cifar-10-batches-py/batches.meta"
    C_VALUES = np.linspace(1e-5, 1e-3, 10)
    TOL = 1e-4

    # (a) Data loading and preprocessing
    loader = DataLoader(DATA_PATHS, TEST_PATH, META_PATH)
    X_train, y_train, X_test, y_test, label_names = loader.load_data()
    
    visualizer = Visualizer()
    # Example image for testing
    visualizer.plot_image(X_train[0], title=label_names[y_train[0]])
    
    preprocessor = Preprocessor()
    X_train_flat = preprocessor.flatten(X_train)
    X_test_flat = preprocessor.flatten(X_test)
    X_train_scaled, X_test_scaled = preprocessor.scale(X_train_flat, X_test_flat)

    # (b) Regularization parameter testing
    print("\nTesting convergence for different C values: ", C_VALUES)
    
    # Note: In Scikit-Learn, C = 1/λ (λ is the regularization strength from the course)
    # Small C (high λ) penalizes coefficients heavily, large C (low λ) reduces regularization
    for C in C_VALUES:
        model = LogisticRegression(
            C=C,
            solver='lbfgs',  # Suitable for L2 regularization and multiclass problems
            max_iter=5000,
            tol=TOL,
            n_jobs=-1
        )
        try:
            model.fit(X_train_scaled, y_train)
            print(f"C={C:.1e}: Converged in {model.n_iter_} iterations")
        except Exception as e:
            print(f"C={C:.1e}: Failed to converge - {str(e)}")

    # (c) Accuracy-based CV
    print("\nCross-validating accuracy")
    # Accuracy prioritizes correct class labels but ignores probabilistic confidence
    cv_model_acc = LogisticRegressionCV(
        Cs=C_VALUES,
        cv=4,
        solver='lbfgs',
        max_iter=5000,
        tol=TOL,
        scoring='accuracy',
        n_jobs=-1   
    )
    cv_model_acc.fit(X_train_scaled, y_train)
    visualizer.plot_scores(C_VALUES, np.mean(cv_model_acc.scores_[True], axis=0), 
                          'Accuracy', '#66EE91')

    # (d) Log loss-based CV
    print("\nCross-validating log loss")
    # Log loss evaluates probabilistic calibration, penalizing overconfident incorrect predictions
    """ da qua in poi ho problemi di convergenza, ci mette troppe iterazioni e troppo tempo, sto lavorando per risolvere"""
    kf = KFold(n_splits=4)
    ll_scores = []
    for C in C_VALUES:
        model = LogisticRegression(
            C=C,
            solver='saga',
            max_iter=10000,
            tol=TOL,
            n_jobs=-1  
        )
        scores = cross_validate(
            model, X_train_scaled, y_train, 
            cv=kf, scoring='neg_log_loss'
        )
        ll_scores.append(-np.mean(scores['test_score']))
    visualizer.plot_scores(C_VALUES, ll_scores, 'Log Loss', '#669FEE')

    # (e) Final model evaluation
    # Prioritizing log loss ensures probabilities are well-calibrated (critical for uncertainty-aware decisions)
    final_C = C_VALUES[np.argmin(ll_scores)]
    final_model = LogisticRegression(
        C=final_C,
        solver='lbfgs',
        max_iter=5000,
        tol=TOL,
        n_jobs=-1
    )
    final_model.fit(X_train_scaled, y_train)

    # Performance reporting
    y_test_pred = final_model.predict(X_test_scaled)
    y_test_proba = final_model.predict_proba(X_test_scaled)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")
    print(f"Test Log Loss: {log_loss(y_test, y_test_proba):.4f}")

if __name__ == "__main__":
    main()