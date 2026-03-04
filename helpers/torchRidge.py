from typing import List
from sklearn.model_selection import KFold
import torch

import matplotlib.pyplot as plt
import numpy as np

class TorchRidge:
    def __init__(self, alpha = 0, fit_intercept = True, device = "cpu"):
        self.alpha = alpha
        if not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha)
        self.alpha.to(device)
        self.fit_intercept = fit_intercept
        self.device = device

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.hstack([torch.ones(X.shape[0], 1), X])

        X = X.to(self.device)
        y = y.to(self.device)

        lhs = X.T @ X
        rhs = X.T @ y
        w = []
        if self.alpha.dim() == 0:
            ridge = self.alpha * torch.eye(lhs.shape[0], device=self.device)
            self.w = torch.linalg.lstsq(lhs + ridge, rhs, driver='gelsy').solution
        else:
            assert self.alpha.shape[0] == y.shape[1], "Number of alphas don't match number of y columns"
            for i in range(y.shape[1]):
                alpha = self.alpha[i]
                ridge = alpha * torch.eye(lhs.shape[0], device=self.device)
                w.append(torch.linalg.lstsq(lhs + ridge, rhs[:, [i]], driver='gelsy').solution)
            self.w = torch.vstack(w).to(self.device)

    def predict(self, X: torch.Tensor) -> None:
        X = X.to(self.device)
        if self.fit_intercept:
            X = torch.hstack([torch.ones(X.shape[0], 1), X])
        X = X.to(self.device)
        return X @ self.w
    
    class RidgePerElectrode:
        """
        Ridge regression with per-electrode hyperparameter tuning.
        Each electrode gets its own optimized alpha via cross-validation.
        """
        def __init__(
            self, 
            alpha_range: List[float] = None,
            n_folds: int = 5,
            fit_intercept: bool = True,
            use_torch: bool = False,
            device: str = "cpu",
            random_state: int = 42
        ):
            """
            Args:
                alpha_range: List of alpha values to try. If None, uses [0.01, 0.1, 1.0, 10.0, 100.0]
                n_folds: Number of CV folds for hyperparameter selection
                fit_intercept: Whether to fit an intercept term
                use_torch: Whether to use PyTorch (TorchRidge) or NumPy
                device: Device for PyTorch ("cpu" or "cuda")
                random_state: Random seed for reproducibility
            """
            self.alpha_range = alpha_range if alpha_range is not None else [0.01, 0.1, 1.0, 10.0, 100.0]
            self.n_folds = n_folds
            self.fit_intercept = fit_intercept
            self.use_torch = use_torch
            self.device = device
            self.random_state = random_state
            
            # These will be set during fitting
            self.best_alphas_ = None
            self.models_ = []
            self.n_electrodes_ = None
            
        def _cross_validate_alpha(self, X: np.ndarray, y: np.ndarray, electrode_idx: int) -> float:
            """
            Use k-fold CV to select the best alpha for a single electrode.
            """
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            best_alpha = self.alpha_range[0]
            best_score = -np.inf
            
            for alpha in self.alpha_range:
                fold_scores = []
                
                for train_idx, val_idx in kf.split(X):
                    X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                    y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                    
                    # Fit model with this alpha
                    if self.use_torch:
                        model = self._fit_torch_single(X_fold_train, y_fold_train, alpha)
                        y_pred = self._predict_torch_single(model, X_fold_val)
                    else:
                        W = self._fit_numpy_single(X_fold_train, y_fold_train, alpha)
                        y_pred = self._predict_numpy_single(X_fold_val, W)
                    
                    # Calculate R^2 score
                    ss_res = np.sum((y_fold_val - y_pred) ** 2)
                    ss_tot = np.sum((y_fold_val - y_fold_val.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot
                    fold_scores.append(r2)
                
                # Average R^2 across folds
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    
            return best_alpha
        
        def _fit_numpy_single(self, X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
            """Fit ridge regression for a single electrode using NumPy."""
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            
            n_features = X.shape[1]
            I = np.eye(n_features)
            W = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)
            return W
        
        def _predict_numpy_single(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
            """Predict for a single electrode using NumPy."""
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            return X @ W
        
        def _fit_torch_single(self, X: np.ndarray, y: np.ndarray, alpha: float):
            """Fit ridge regression for a single electrode using PyTorch."""
            X_torch = torch.tensor(X, dtype=torch.float32)
            y_torch = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            
            model = TorchRidge(alpha=alpha, fit_intercept=self.fit_intercept, device=self.device)
            model.fit(X_torch, y_torch)
            return model
        
        def _predict_torch_single(self, model, X: np.ndarray) -> np.ndarray:
            """Predict for a single electrode using PyTorch."""
            X_torch = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                pred = model.predict(X_torch).cpu().numpy().flatten()
            return pred
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgePerElectrode':
            """
            Fit the model with per-electrode alpha selection.
            
            Args:
                X: Feature matrix (n_samples, n_features)
                y: Target matrix (n_samples, n_electrodes)
            """
            self.n_electrodes_ = y.shape[1]
            self.best_alphas_ = []
            self.models_ = []
            
            print(f"Fitting {self.n_electrodes_} electrodes with cross-validation...")
            
            for e in range(self.n_electrodes_):
                y_e = y[:, e]
                
                # Cross-validate to find best alpha
                best_alpha = self._cross_validate_alpha(X, y_e, e)
                self.best_alphas_.append(best_alpha)
                
                # Fit final model with best alpha on full training data
                if self.use_torch:
                    model = self._fit_torch_single(X, y_e, best_alpha)
                    self.models_.append(model)
                else:
                    W = self._fit_numpy_single(X, y_e, best_alpha)
                    self.models_.append(W)
                
                if (e + 1) % 10 == 0:
                    print(f"  Completed {e + 1}/{self.n_electrodes_} electrodes")
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Predict for all electrodes.
            
            Args:
                X: Feature matrix (n_samples, n_features)
                
            Returns:
                Predictions (n_samples, n_electrodes)
            """
            predictions = []
            
            for e in range(self.n_electrodes_):
                if self.use_torch:
                    y_pred = self._predict_torch_single(self.models_[e], X)
                else:
                    y_pred = self._predict_numpy_single(X, self.models_[e])
                predictions.append(y_pred)
            
            return np.column_stack(predictions)
        
        def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
            """
            Evaluate the model and return metrics per electrode.
            
            Args:
                X: Feature matrix (n_samples, n_features)
                y: Target matrix (n_samples, n_electrodes)
                
            Returns:
                Dictionary with MSE and R^2 per electrode
            """
            y_pred = self.predict(X)
            
            mse_list = []
            r2_list = []
            
            for e in range(self.n_electrodes_):
                y_true = y[:, e]
                y_pred_e = y_pred[:, e]
                
                # MSE
                mse = np.mean((y_true - y_pred_e) ** 2)
                mse_list.append(mse)
                
                # R^2
                ss_res = np.sum((y_true - y_pred_e) ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot
                r2_list.append(r2)
            
            return {
                'mse': mse_list,
                'r2': r2_list
            }

