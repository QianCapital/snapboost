import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, r2_score
from sklearn.utils.validation import check_is_fitted
from .utils import MeanSquaredError, Logistic


def _normalize_classification_labels(y):
    """Convert 0/1 labels to -1/+1 for logistic loss."""
    y = np.asarray(y, dtype=float).ravel()
    if np.any(np.isnan(y)):
        raise ValueError("Classification labels must not contain NaN values.")
    unique = np.unique(y)
    if np.array_equal(unique, [-1.0, 1.0]) or np.array_equal(unique, [-1.0]) or np.array_equal(unique, [1.0]):
        return y
    if np.all(np.isin(unique, [0.0, 1.0])):
        return np.where(y == 0, -1.0, 1.0)
    raise ValueError("Classification labels must be 0/1 or -1/+1.")


def _labels_for_log_loss(y):
    """Convert labels to 0/1 for sklearn's log_loss."""
    y = np.asarray(y, dtype=float).ravel()
    if np.any(np.isnan(y)):
        raise ValueError("Classification labels must not contain NaN values.")
    unique = np.unique(y)
    if np.all(np.isin(unique, [0.0, 1.0])):
        return y
    if np.all(np.isin(unique, [-1.0, 1.0])):
        return np.where(y == -1, 0.0, 1.0)
    raise ValueError("Classification labels must be 0/1 or -1/+1.")


def _validate_X(X):
    """Validate and reshape feature matrix."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    elif X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    return X


def _validate_X_y(X, y):
    """Validate feature matrix and label vector shapes."""
    X = _validate_X(X)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    elif y.ndim != 1:
        raise ValueError(f"y must be a 1D array, got shape {y.shape}.")
    if y.shape[0] == 0:
        raise ValueError("y must contain at least one label.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have inconsistent lengths: {X.shape[0]} vs {y.shape[0]}."
        )
    return X, y


class HNBM(BaseEstimator):
    """
    Heterogeneous Newton Boosting Machine
    Args:
        num_iterations (int): number of boosting iterations
        learning_rate (float): learning rate
        mode (string): classification or regression
        random_state (int): random seed for base learner selection
        verbose (bool): whether to show a progress bar during training
    Attributes:
        ensemble_ (list): Ensemble after training
    """
    def __init__(self, num_iterations=100, learning_rate=0.1, mode="classification",
                 random_state=None, verbose=True):
        if mode not in ("classification", "regression"):
            raise ValueError("Invalid mode: specify 'classification' or 'regression'.")
        if num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {num_iterations}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.mode = mode
        self.random_state = random_state
        self.verbose = verbose
        self.base_learners_ = []
        self.probabilities_ = []
        self.ensemble_ = []

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = (
            "classifier" if self.mode == "classification" else "regressor"
        )
        return tags

    def _check_fitted(self):
        check_is_fitted(self, "ensemble_")
        if not self.ensemble_:
            raise NotFittedError(
                "This HNBM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

    def set_params(self, **params):
        result = super().set_params(**params)
        if params:
            if "num_iterations" in params and self.num_iterations < 1:
                raise ValueError(
                    f"num_iterations must be >= 1, got {self.num_iterations}."
                )
            if "learning_rate" in params and self.learning_rate <= 0:
                raise ValueError(
                    f"learning_rate must be > 0, got {self.learning_rate}."
                )
            if "mode" in params and self.mode not in ("classification", "regression"):
                raise ValueError(
                    "Invalid mode: specify 'classification' or 'regression'."
                )
            self.ensemble_ = []
        return result

    def fit(self, X, y):
        """
        Train the model
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
        Returns:
            self
        """
        if not self.base_learners_:
            raise ValueError(
                "No base learners configured. Use SnapBoost or add learners to base_learners_."
            )

        X, y = _validate_X_y(X, y)
        if self.mode == "classification":
            y = _normalize_classification_labels(y)
            self.classes_ = np.array([0, 1])
        else:
            y = np.asarray(y, dtype=float).ravel()

        rng = np.random.default_rng(self.random_state)
        z = np.zeros(X.shape[0])
        self.ensemble_ = []
        iterations = range(self.num_iterations)
        if self.verbose:
            iterations = tqdm(iterations, desc="Training")

        for _ in iterations:
            g, h = self.loss_.compute_derivatives(y, z)
            idx = rng.choice(len(self.base_learners_), p=self.probabilities_)
            base_learner = clone(self.base_learners_[idx])
            base_learner.fit(X, -np.divide(g, h), sample_weight=h)
            z += base_learner.predict(X) * self.learning_rate
            self.ensemble_.append(base_learner)

        return self

    @property
    def loss_(self):
        return Logistic if self.mode == "classification" else MeanSquaredError

    @property
    def num_iterations_(self):
        return self.num_iterations

    @property
    def learning_rate_(self):
        return self.learning_rate

    def _raw_predict(self, X):
        """Return raw model output (logits for classification, values for regression)."""
        self._check_fitted()
        X = _validate_X(X)
        preds = np.zeros(X.shape[0])
        for learner in self.ensemble_:
            preds += self.learning_rate * learner.predict(X)
        return preds

    def decision_function(self, X):
        """
        Return classification logits.
        Args:
            X (np.ndarray): Feature matrix
        """
        if self.mode != "classification":
            raise ValueError("decision_function is only available in classification mode.")
        return self._raw_predict(X)

    def predict(self, X):
        """
        Predict using the model.
        Classification returns 0/1 labels; regression returns continuous values.
        Args:
            X (np.ndarray): Feature matrix
        """
        if self.mode == "classification":
            logits = self.decision_function(X)
            return (logits >= 0).astype(int)
        return self._raw_predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities (classification mode only).
        Args:
            X (np.ndarray): Feature matrix
        Returns:
            np.ndarray of shape (n_samples, 2) with [P(y=0), P(y=1)]
        """
        if self.mode != "classification":
            raise ValueError("predict_proba is only available in classification mode.")
        logits = self.decision_function(X)
        prob_pos = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - prob_pos, prob_pos])

    def score(self, X, y):
        """
        Return the default score for the model mode (accuracy or R^2).
        """
        self._check_fitted()
        _, y = _validate_X_y(X, y)
        if self.mode == "classification":
            y = _labels_for_log_loss(y)
            return accuracy_score(y, self.predict(X))
        y = np.asarray(y, dtype=float).ravel()
        return r2_score(y, self.predict(X))

    def evaluate(self, X, y):
        """
        Evaluate trained model
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
        """
        self._check_fitted()
        if self.mode == "classification":
            _, y = _validate_X_y(X, y)
            y = _labels_for_log_loss(y)
            prob_pos = self.predict_proba(X)[:, 1]
            loss = log_loss(y, prob_pos, labels=[0, 1])
            print("Log Loss: %.4f" % loss)
        else:
            _, y = _validate_X_y(X, y)
            y = np.asarray(y, dtype=float).ravel()
            preds = self._raw_predict(X)
            loss = np.sqrt(mean_squared_error(y, preds))
            print("RMSE: %.4f" % loss)
        return loss
