import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict, KFold, cross_validate, GridSearchCV
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
        plt.yscale('log')  # Added for log-log plot
        plt.xlabel('C (Regularization Strength)')
        plt.ylabel(score_name)
        plt.title(f'{score_name} vs. Regularization Strength (log-log)')
        plt.grid(True, which='both', linestyle='--')
        plt.show()
        # Note: Log-log plots may not be ideal for accuracy (bounded [0,1]), 
        # but it aligns with the question's explicit requirement.

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
    C_VALUES = np.logspace(-11, -3, 9)  #considering high values of C bring the optimization to not converge 
    #print("C values: ", C_VALUES)
    #--> strong regularization required
    TOL = 1e-3

    # (a) Data loading and preprocessing
    loader = DataLoader(DATA_PATHS, TEST_PATH, META_PATH)
    X_train, y_train, X_test, y_test, label_names = loader.load_data()
    
    visualizer = Visualizer()
    visualizer.plot_image(X_train[0], title=label_names[y_train[0]])
    
    preprocessor = Preprocessor()
    X_train_flat = preprocessor.flatten(X_train)
    X_test_flat = preprocessor.flatten(X_test)
    X_train_scaled, X_test_scaled = preprocessor.scale(X_train_flat, X_test_flat)

    # (b) Regularization parameter testing (updated solver and parameters)
    def train_model(C):
        model = LogisticRegression(
            C=C,
            solver='lbfgs',       # Standardized solver
            max_iter=5000,       # Align with CV settings
            tol=TOL,
            n_jobs=1              # Avoid nested parallelism
        )
        try:
            model.fit(X_train_scaled, y_train)
            print(f"C={C:.1e}: Converged in {model.n_iter_} iterations")
        except Exception as e:
            print(f"C={C:.1e}: Failed to converge - {str(e)}")
        return model

    print("\nTraining models in parallel for different C values: ", C_VALUES)
    models = Parallel(n_jobs=-1)(delayed(train_model)(C) for C in C_VALUES)

    # (c) Accuracy-based CV
    print("\nCross-validating accuracy")
    cv_model_acc = LogisticRegressionCV(
        Cs=C_VALUES,
        cv=4,
        solver='lbfgs',
        max_iter=5000,
        tol=TOL,
        scoring='accuracy',
        n_jobs=1                  # Avoid nested parallelism
    )
    cv_model_acc.fit(X_train_scaled, y_train)
    visualizer.plot_scores(C_VALUES, np.mean(cv_model_acc.scores_[True], axis=0), 
                          'Accuracy', '#66EE91')

    # (d) Log loss-based CV (corrected with proper cross-validation)
    print("\nCross-validating log loss")
    ll_scores = []
    for C in C_VALUES:
        model = LogisticRegression(
            C=C,
            solver='lbfgs',
            max_iter=5000,
            tol=TOL,
            n_jobs=1  
        )
        probas = cross_val_predict(
            model, 
            X_train_scaled, 
            y_train,
            cv=KFold(n_splits=4),
            method='predict_proba',
            n_jobs=-1
        )
        ll_scores.append(log_loss(y_train, probas))
    visualizer.plot_scores(C_VALUES, ll_scores, 'Log Loss', '#669FEE')

    # (e) Final model evaluation (added training metrics)
    final_C = C_VALUES[np.argmin(ll_scores)]
    final_model = LogisticRegression(
        C=final_C,
        solver='lbfgs',
        max_iter=5000,
        tol=TOL,
        n_jobs=1
    )
    final_model.fit(X_train_scaled, y_train)

    # Training metrics
    y_train_pred = final_model.predict(X_train_scaled)
    y_train_proba = final_model.predict_proba(X_train_scaled)
    print(f"\nTraining Accuracy: {accuracy_score(y_train, y_train_pred):.2%}")
    print(f"Training Log Loss: {log_loss(y_train, y_train_proba):.4f}")

    # Test metrics
    y_test_pred = final_model.predict(X_test_scaled)
    y_test_proba = final_model.predict_proba(X_test_scaled)
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.2%}")
    print(f"Test Log Loss: {log_loss(y_test, y_test_proba):.4f}")

if __name__ == "__main__":
    main()
