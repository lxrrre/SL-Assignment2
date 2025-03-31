import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter("always", ConvergenceWarning)

class A2_cifar10:
    def __init__(self):
        self.solver = 'lbfgs'
        self.verbose = 0
        self.scaler = StandardScaler()
        self.folds= 4
        self.subset_frac = 0.2  # Portion of the full dataset to use
        self.C_values = np.logspace(-10, -2, 9)
        #self.C_values = np.linspace(3e-4, 1.5e-3, 10)  

    def unpickle(self, file):
        """Load CIFAR-10 data from pickle file"""
        import pickle
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='latin1')

    def train_and_evaluate_model(self, C_value, X_train, y_train, X_test, y_test):
        """Helper function to train and evaluate a logistic regression model"""
        model = LogisticRegression(
            max_iter=2000,
            C=C_value,
            solver=self.solver,
            verbose=self.verbose
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Number of iterations: {model.n_iter_}")
        return model

    def run_cross_validation(self, X_train_scaled, y_train_small, scoring='accuracy'):
        """Run cross-validated logistic regression with specified scoring"""
        cv_model = LogisticRegressionCV(
            max_iter=2000,
            Cs=self.C_values,
            cv=self.folds,
            solver=self.solver,
            n_jobs=-1,
            verbose=self.verbose,
            scoring=scoring
        )
        cv_model.fit(X_train_scaled, y_train_small)
        
        # Plot results
        mean_scores = np.mean(cv_model.scores_[0], axis=0)
        plt.figure(figsize=(8, 5))
        plt.loglog(cv_model.Cs_, mean_scores, marker='o', linestyle='-')  
        plt.xticks(cv_model.Cs_, labels=[f"{c:.2e}" for c in cv_model.Cs_], rotation=90)
        plt.xlabel('Regularization parameter C')
        plt.ylabel('Score' if scoring == 'accuracy' else 'Log Loss')
        plt.grid(True, which='both')
        plt.show()
        return cv_model

    def manual_log_loss_cv(self, X_train_scaled, y_train_small):
        """Manual cross-validation with log loss scoring"""
        kfold = StratifiedKFold(n_splits=self.folds)
        log_losses = []
        
        for C in self.C_values:
            fold_scores = []
            for train_idx, val_idx in kfold.split(X_train_scaled, y_train_small):
                X_tr, y_tr = X_train_scaled[train_idx], y_train_small[train_idx]
                X_val, y_val = X_train_scaled[val_idx], y_train_small[val_idx]
                
                model = LogisticRegression(
                    max_iter=2000,
                    C=C,
                    solver=self.solver,
                    verbose=self.verbose
                )
                model.fit(X_tr, y_tr)
                proba = model.predict_proba(X_val)
                fold_scores.append(log_loss(y_val, proba))
                
            log_losses.append(np.mean(fold_scores))
        
        plt.figure(figsize=(8, 5))
        plt.loglog(self.C_values, log_losses, marker='o', linestyle='-')
        plt.xticks(self.C_values, labels=[f"{c:.2e}" for c in self.C_values], rotation=90)
        plt.xlabel('Regularization parameter C')
        plt.ylabel('Log Loss')
        plt.grid(True, which='both')
        plt.show()
        
        best_C = self.C_values[np.argmin(log_losses)]
        return best_C

    def make_final_model(self, C_value,X_train_full, y_train):
        final_model = LogisticRegression(
        max_iter=2000,
        C = C_value,
        solver= self.solver,
        verbose=self.verbose
        )
        final_model.fit(X_train_full, y_train)
        return final_model
    
    def print_metrics(self, model, X_train, X_test, y_train, y_test):
        """Print performance metrics for a trained model"""
        for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
            acc = accuracy_score(y, model.predict(X))
            loss = log_loss(y, model.predict_proba(X))
            print(f"{name} Accuracy: {acc:.4f}")
            print(f"{name} Log-Loss: {loss:.4f}")


def main():
    # Execution time
    load_start = time.time()

    # Load data
    model_instance = A2_cifar10()
    train_batches = [model_instance.unpickle(f"./cifar-10-batches-py/data_batch_{i}") for i in range(1,6)]
    test_batch = model_instance.unpickle("./cifar-10-batches-py/test_batch")
    meta_data = model_instance.unpickle("./cifar-10-batches-py/batches.meta")

    # Prepare data
    X_train = np.concatenate([batch["data"] for batch in train_batches])
    y_train = np.concatenate([batch["labels"] for batch in train_batches])
    X_test, y_test = test_batch["data"], test_batch["labels"]

    # Create smaller subset for development
    train_subset_size, test_subset_size = int(len(X_train) * model_instance.subset_frac), int(len(X_test) * model_instance.subset_frac)
    X_train_small, y_train_small = X_train[:train_subset_size], y_train[:train_subset_size]
    X_test_small, y_test_small = X_test[:test_subset_size], y_test[:test_subset_size]

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_small)
    X_test_scaled = scaler.transform(X_test_small)

    # Initial model training
    print("Training base model:\nDataset train size: ", len(X_train_small), "\nDataset test size: ", len(X_test_small),"\n")
    base_model = model_instance.train_and_evaluate_model(1e-3, X_train_scaled, y_train_small, X_test_scaled, y_test_small)

    # Cross-validated model training
    print("\nRunning cross-validation with accuracy scoring:")
    cv_model_acc = model_instance.run_cross_validation(X_train_scaled, y_train_small, scoring='accuracy')
    best_C_acc = cv_model_acc.C_[0]
    mean_scores_acc = np.mean(cv_model_acc.scores_[0], axis=0)
    best_C_acc_idx = np.argmin(np.abs(model_instance.C_values - best_C_acc))
    best_acc_value = mean_scores_acc[best_C_acc_idx]
    print(f"Best C (accuracy): {best_C_acc:.2e} with accuracy {best_acc_value:.3f}")

    print("\nRunning cross-validation with log loss scoring:")
    best_C_log_loss = model_instance.manual_log_loss_cv(X_train_scaled, y_train_small)
    best_C_log_loss_idx = np.argmin(np.abs(model_instance.C_values - best_C_log_loss))
    best_log_loss_accuracy = mean_scores_acc[best_C_log_loss_idx]
    print(f"Best C (log loss): {best_C_log_loss:.2e} with accuracy {best_log_loss_accuracy:.3f}")

    # Full dataset training
    print("\nTraining final model on full dataset:")
    scaler_full = StandardScaler()
    X_train_full = scaler_full.fit_transform(X_train)
    X_test_full = scaler_full.transform(X_test)

    # Final model training
    final_model_1 = model_instance.make_final_model(best_C_acc, X_train_full, y_train) 

    # Evaluate final model
    if best_C_acc == best_C_log_loss:
            print("\nEvaluating final models based on best regularization parameter (is the same for accuracy and log loss):")
            model_instance.print_metrics(final_model_1, X_train_full, X_test_full, y_train, y_test)
    else:   
        final_model_2 = model_instance.make_final_model(best_C_log_loss, X_train_full, y_train)

        print("\nEvaluating final models based on best regularization parameter for accuracy:")
        model_instance.print_metrics(final_model_1, X_train_full, X_test_full, y_train, y_test)
        print("\nEvaluating final models based on best regularization parameter for log loss:")
        model_instance.print_metrics(final_model_2, X_train_full, X_test_full, y_train, y_test)

    total_time = time.time() - load_start
    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print("All tasks completed!")

if __name__ == "__main__":
    main()
