import numpy as np
import pandas as pd
from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sktime.transformations.panel.rocket import MiniRocketMultivariate, MultiRocketMultivariate, Rocket
from sktime.transformations.panel.padder import PaddingTransformer
from src.models.base import IClassifier

import logging

logger = logging.getLogger(__name__)

class BaseRocketXGBClassifier(IClassifier):
    """Base class for ROCKET variants combined with XGBoost."""
    def __init__(self, rocket_transformer, n_estimators: int = 100, pad_length: Optional[int] = None):
        """
        Initializes the pipeline with a ROCKET transformer and an XGBoost classifier.
        
        Args:
            rocket_transformer: An instantiated sktime ROCKET transformer.
            n_estimators: Number of trees for XGBoost.
            pad_length: If provided, adds a PaddingTransformer to the pipeline.
        """
        self.xgb_clf = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            n_estimators=n_estimators
        )
        
        steps = []
        if pad_length is not None:
            steps.append(('padder', PaddingTransformer(pad_length=pad_length, fill_value=0)))
        else:
            steps.append(('padder', PaddingTransformer(fill_value=0)))
            
        steps.append(('rocket', rocket_transformer))
        steps.append(('xgb', self.xgb_clf))
        
        self.pipeline = Pipeline(steps)

    def fit(self, X_train, y_train) -> None:
        """Trains the ROCKET+XGB pipeline on pre-segmented patterns."""
        if isinstance(X_train, pd.DataFrame) and isinstance(X_train.index, pd.MultiIndex):
            self.max_len_ = X_train.groupby(level=0).size().max()
        elif isinstance(X_train, np.ndarray) and X_train.ndim == 3:
            self.max_len_ = X_train.shape[2]
        else:
            self.max_len_ = None
        self.pipeline.fit(X_train, y_train)

    def _truncate(self, X):
        if getattr(self, 'max_len_', None) is None:
            return X
            
        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
            lengths = X.groupby(level=0).size()
            if lengths.max() > self.max_len_:
                X = X.groupby(level=0).head(self.max_len_)
        elif isinstance(X, np.ndarray) and X.ndim == 3:
            if X.shape[2] > self.max_len_:
                X = X[:, :, :self.max_len_]
        return X

    def predict_proba(self, X) -> np.ndarray:
        """Returns probabilities for the classes."""
        X = self._truncate(X)
        return self.pipeline.predict_proba(X)
        
    def predict(self, X) -> np.ndarray:
        """Returns the predicted classes."""
        X = self._truncate(X)
        return self.pipeline.predict(X)

class RocketXGBClassifier(BaseRocketXGBClassifier):
    """ROCKET combined with XGBoost."""
    def __init__(self, num_kernels: int = 10000, n_estimators: int = 100, pad_length: Optional[int] = None):
        rocket = Rocket(num_kernels=num_kernels)
        super().__init__(rocket, n_estimators, pad_length)

class MiniRocketXGBClassifier(BaseRocketXGBClassifier):
    """MiniROCKET combined with XGBoost."""
    def __init__(self, num_kernels: int = 10000, n_jobs: int = 1, n_estimators: int = 100, pad_length: Optional[int] = None):
        mini_rocket = MiniRocketMultivariate(num_kernels=num_kernels, n_jobs=n_jobs)
        super().__init__(mini_rocket, n_estimators, pad_length)

class MultiRocketXGBClassifier(BaseRocketXGBClassifier):
    """MultiROCKET combined with XGBoost."""
    def __init__(self, num_kernels: int = 10000, n_jobs: int = 1, n_estimators: int = 100, pad_length: Optional[int] = None):
        multi_rocket = MultiRocketMultivariate(num_kernels=num_kernels, n_jobs=n_jobs)
        super().__init__(multi_rocket, n_estimators, pad_length)
