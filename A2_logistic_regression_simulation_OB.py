"""
@author: Bracci Lorenzo - Musazzi Federica - Schiavi Francesco
"""

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegressionSimulation:
    def __init__(self):
        """
        Initialize simulation parameters and storage attributes
        """
        # Simulation configuration
        self.default_n = 1000        # Default sample size
        self.beta_star = np.array([0.2, -0.8])  # True beta values
        self.mu1 = [-0.5, 1]        # Mean for first feature cluster
        self.mu2 = [2, -1]           # Mean for second feature cluster
        self.cov = [[1, 0.7], [0.7, 1]]  # Shared covariance matrix
        self.S = 1000                # Number of Monte Carlo simulations
        self.nr_steps = 1000         # Max Newton-Raphson iterations
        self.tolerance = 1e-10       # Convergence threshold
        
        # Visualization parameters
        self.contour_resolution = 100  # Grid points for contour plot
        self.hist_bins = 100           # Bins for MLE histograms
        
        # Data storage attributes
        self.simulated_features = None  # Feature matrix
        self.simulated_labels = None    # Label vector
        self.beta_sequence = None      # NR iteration history

    def simulate_features(self, n):
        """Generate multivariate normal features for given sample size"""
        half = n//2
        np.random.seed(1)  # Ensure reproducibility
        x1 = np.random.multivariate_normal(self.mu1, self.cov, half)
        x2 = np.random.multivariate_normal(self.mu2, self.cov, half)
        return np.vstack((x1, x2)).astype(np.float64)

    def logistic(self, x):
        """Logistic sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def generate_labels(self, features, beta):
        """Simulate binary labels using logistic model"""
        p = self.logistic(features @ beta)
        return np.array([np.random.binomial(1, pi) for pi in p])

    def newton_raphson(self, features, target):
        """Perform Newton-Raphson optimization for logistic regression"""
        beta = np.zeros(features.shape[1])
        for _ in range(self.nr_steps):
            p = self.logistic(features @ beta)
            W = np.diag(p * (1 - p))
            gradient = features.T @ (target - p)
            
            if np.linalg.norm(gradient) > self.tolerance:
                Hessian = -features.T @ W @ features
                beta -= np.linalg.solve(Hessian, gradient)
            else:
                break
        return beta

    def newton_raphson_with_history(self, features, target):
        """NR implementation that stores beta iteration history"""
        beta = np.zeros(features.shape[1])
        history = [beta.copy()]
        for _ in range(self.nr_steps):
            p = self.logistic(features @ beta)
            W = np.diag(p * (1 - p))
            gradient = features.T @ (target - p)
            
            if np.linalg.norm(gradient) > self.tolerance:
                Hessian = -features.T @ W @ features
                beta -= np.linalg.solve(Hessian, gradient)
                history.append(beta.copy())
            else:
                break
        return np.array(history)

    def run_monte_carlo(self, sample_sizes):
        """Run simulation study for different sample sizes"""
        for n in sample_sizes:
            features = self.simulate_features(n)
            mle_estimates = np.zeros((self.S, 2))
            
            for s in range(self.S):
                labels = self.generate_labels(features, self.beta_star)
                mle_estimates[s] = self.newton_raphson(features, labels)
            
            beta_mean = np.mean(mle_estimates, axis=0)
            norm_diff = np.linalg.norm(beta_mean - self.beta_star)
            
            print(f"\n=== Results for n={n} ===")
            print(f"Mean beta estimate: [{beta_mean[0]:.6f}, {beta_mean[1]:.6f}]")
            print(f"Average distance from true beta: {norm_diff:.6f}")
            
            # Plot histograms
            plt.figure()
            plt.hist(mle_estimates, bins=self.hist_bins)
            plt.title(f'MLE Distribution (n={n})')
            plt.xlabel('Estimated Beta Value', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.savefig(f'mle_distribution_n_{n}.png')
            plt.close()

    def plot_feature_scatter(self):
        """Create scatter plot of features colored by labels"""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.simulated_features[:, 0], 
                             self.simulated_features[:, 1],
                             c=self.simulated_labels, alpha=0.5)
        
        # Create legend handles
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=scatter.cmap(scatter.norm(0)), 
                    markersize=10, label='0'),
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=scatter.cmap(scatter.norm(1)), 
                    markersize=10, label='1')
        ]
        plt.legend(handles, ['0', '1'], title="Labels")
        plt.xlabel('Feature 1', fontsize=12)
        plt.ylabel('Feature 2', fontsize=12)
        plt.savefig('feature_scatter.png')
        plt.close()

    def plot_loglikelihood_contour(self):
        """Create contour plot of log-likelihood surface"""
        # Grid setup
        b1 = np.linspace(-0.05, 0.3, self.contour_resolution)
        b2 = np.linspace(-1, 0.1, self.contour_resolution)
        B1, B2 = np.meshgrid(b1, b2)
        
        # Calculate log-likelihood values
        loglik = np.array([self._log_likelihood(np.array([bv1, bv2])) 
                          for bv1, bv2 in zip(B1.ravel(), B2.ravel())])
        loglik = loglik.reshape(B1.shape)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.contourf(B1, B2, loglik, levels=50, cmap='viridis')
        plt.colorbar(label='Log-Likelihood Value')
        
        # Add optimization path and true beta
        plt.plot(self.beta_sequence[:, 0], self.beta_sequence[:, 1], 
                'r.-', label='NR Path')
        plt.plot(*self.beta_star, 'b*', markersize=10, label='True Beta')
        plt.legend()
        plt.xlabel('Beta 1', fontsize=12)
        plt.ylabel('Beta 2', fontsize=12)
        plt.savefig('loglikelihood_contour.png')
        plt.close()

    def _log_likelihood(self, beta):
        """Helper function to calculate log-likelihood"""
        p = self.logistic(self.simulated_features @ beta)
        return np.sum(self.simulated_labels * np.log(p) + 
                    (1 - self.simulated_labels) * np.log(1 - p))

if __name__ == "__main__":
    # Initialize simulation environment
    sim = LogisticRegressionSimulation()
    
    # Generate and plot initial dataset
    sim.simulated_features = sim.simulate_features(sim.default_n)
    sim.simulated_labels = sim.generate_labels(sim.simulated_features, sim.beta_star)
    sim.plot_feature_scatter()
    
    # Single NR estimation
    beta_est = sim.newton_raphson(sim.simulated_features, sim.simulated_labels)
    print("\n=== Initial Estimation ===")
    print(f"Estimated beta: {beta_est}")
    print(f"Distance from true beta: {np.linalg.norm(beta_est - sim.beta_star):.6f}")
    
    # Monte Carlo study
    sim.run_monte_carlo(sample_sizes=[100, 1000])
    
    # Log-likelihood contour plot
    sim.beta_sequence = sim.newton_raphson_with_history(sim.simulated_features, 
                                                      sim.simulated_labels)
    sim.plot_loglikelihood_contour()