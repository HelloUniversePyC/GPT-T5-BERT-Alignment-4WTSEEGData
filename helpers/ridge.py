from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from scipy.stats import pearsonr
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RidgePerElectrode:
    def __init__(self, alpha_range=None, n_folds=5, random_state=42, n_jobs=-1, 
                 compute_pvalues=False, n_permutations=100):
        if alpha_range is None:
            alpha_range = np.logspace(-3, 1, 10)
        
        self.alpha_range = alpha_range
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.compute_pvalues = compute_pvalues
        self.n_permutations = n_permutations
        
        self.models_ = None
        self.best_alphas_ = None
        self.cv_scores_ = None
        self.pvalues_ = None
        
        # Store training data for permutation tests
        self.X_train_ = None
        self.y_train_ = None
    
    def _fit_single_electrode(self, X, y_electrode, electrode_idx):
        ridge_cv = RidgeCV(
            alphas=self.alpha_range,
            cv=self.n_folds,
            scoring='r2',
        )
        ridge_cv.fit(X, y_electrode)
        
        best_alpha = ridge_cv.alpha_
        best_score = ridge_cv.best_score_
        
        return {
            'model': ridge_cv,
            'best_alpha': best_alpha,
            'cv_score': best_score,
            'electrode_idx': electrode_idx
        }
    
    def _permutation_test_single_electrode(self, X, y_electrode, model, electrode_idx):
        """Compute p-value via permutation test for a single electrode."""
        # Get actual R² score
        y_pred = model.predict(X)
        actual_r2 = r2_score(y_electrode, y_pred)
        
        # Permutation test
        rng = np.random.RandomState(self.random_state + electrode_idx)
        null_r2_scores = []
        
        for _ in range(self.n_permutations):
            # Shuffle y labels
            y_permuted = rng.permutation(y_electrode)
            
            # Fit model on permuted data
            model_perm = Ridge(alpha=model.alpha_)
            model_perm.fit(X, y_permuted)
            
            # Predict and compute R²
            y_pred_perm = model_perm.predict(X)
            null_r2 = r2_score(y_permuted, y_pred_perm)
            null_r2_scores.append(null_r2)
        
        # Compute p-value
        null_r2_scores = np.array(null_r2_scores)
        p_value = np.mean(null_r2_scores >= actual_r2)
        return {
            'electrode_idx': electrode_idx,
            'p_value': p_value,
            'actual_r2': actual_r2
        }
    
    def fit(self, X, y):
        _, n_electrodes = y.shape
        
        # Store training data for later use
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        print(f"Fitting {n_electrodes} electrodes with RidgeCV using {self.n_jobs} jobs...")
        
        # Parallel fitting with progress bar
        results = Parallel(n_jobs=self.n_jobs, backend='loky')(
            delayed(self._fit_single_electrode)(X, y[:, i], i)
            for i in tqdm(range(n_electrodes), desc="Fitting electrodes")
        )
        
        # Sort results by electrode index
        results = sorted(results, key=lambda x: x['electrode_idx'])
        
        # Extract results
        self.models_ = [r['model'] for r in results]
        self.best_alphas_ = np.array([r['best_alpha'] for r in results])
        self.cv_scores_ = np.array([r['cv_score'] for r in results])
        
        print(f"  ✓ Completed fitting all {n_electrodes} electrodes")
        
        # Compute p-values if requested
        if self.compute_pvalues:
            print(f"\nComputing p-values via permutation test ({self.n_permutations} permutations)...")
            pvalue_results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(self._permutation_test_single_electrode)(X, y[:, i], self.models_[i], i)
                for i in tqdm(range(n_electrodes), desc="Computing p-values")
            )
            
            # Sort and extract p-values
            pvalue_results = sorted(pvalue_results, key=lambda x: x['electrode_idx'])
            self.pvalues_ = np.array([r['p_value'] for r in pvalue_results])
            print(f"  ✓ Completed p-value computation")
        
        return self
    
    def predict(self, X):
        if self.models_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        
        return predictions
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        n_electrodes = y.shape[1]
        
        mse_scores = np.zeros(n_electrodes)
        r2_scores = np.zeros(n_electrodes)
        pearson_r_scores = np.zeros(n_electrodes)
        
        for i in range(n_electrodes):
            mse_scores[i] = mean_squared_error(y[:, i], predictions[:, i])
            r2_scores[i] = r2_score(y[:, i], predictions[:, i])
            r, _ = pearsonr(y[:, i], predictions[:, i])
            pearson_r_scores[i] = r
        
        return {
            'mse': mse_scores,
            'r2': r2_scores,
            'pearson_r': pearson_r_scores,
            'predictions': predictions
        }
    
    def get_cv_summary(self):
        if self.cv_scores_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        summary = {
            'cv_r2_per_electrode': self.cv_scores_,
            'overall_mean': np.mean(self.cv_scores_),
            'overall_std': np.std(self.cv_scores_)
        }
        
        if self.pvalues_ is not None:
            summary['pvalues_per_electrode'] = self.pvalues_
            summary['n_significant_05'] = np.sum(self.pvalues_ < 0.05)
            summary['n_significant_01'] = np.sum(self.pvalues_ < 0.01)
        
        return summary