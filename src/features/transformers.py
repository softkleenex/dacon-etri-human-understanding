import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


class LeakFreeSubjectZScaler(BaseEstimator, TransformerMixin):
    """
    A strictly leak-free subject-wise Z-Score Normalization transformer
    that fits mean and std statistics ONLY on the train folds
    and applies them to any input data without future look-ahead.
    """

    def __init__(self, id_col="subject_id", numeric_cols=None):
        self.id_col = id_col
        self.numeric_cols = numeric_cols
        self.means_ = None
        self.stds_ = None

    def fit(self, X, y=None):
        if self.numeric_cols is None:
            exclude_cols = {self.id_col, "sleep_date", "lifelog_date", "date", "timestamp"}
            self.numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        # Fit statistics strictly on X (assumed to be Train fold only)
        self.means_ = X.groupby(self.id_col)[self.numeric_cols].mean()
        self.stds_ = X.groupby(self.id_col)[self.numeric_cols].std().replace(0.0, np.nan).fillna(1.0)
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Vectorized mapping to prevent Pandas fragmentation warnings
        mapped_means = self.means_.reindex(X_copy[self.id_col]).fillna(0.0).values
        mapped_stds = self.stds_.reindex(X_copy[self.id_col]).fillna(1.0).values

        # Calculate subject-specific Z-Score
        zscore_vals = (X_copy[self.numeric_cols].values - mapped_means) / (mapped_stds + 1e-6)

        # Concatenate Z-Score features
        zscore_df = pd.DataFrame(
            zscore_vals, columns=[f"{col}_subject_zscore" for col in self.numeric_cols], index=X_copy.index
        )
        return pd.concat([X_copy, zscore_df], axis=1)


class FoldWisePCA(BaseEstimator, TransformerMixin):
    """
    A strictly leak-free PCA dimensionality reduction transformer
    that fits a SimpleImputer and PCA model ONLY on the train folds.
    """

    def __init__(self, n_components=5, random_state=42, numeric_cols=None):
        self.n_components = n_components
        self.random_state = random_state
        self.numeric_cols = numeric_cols
        self.imputer = None
        self.pca = None

    def fit(self, X, y=None):
        if self.numeric_cols is None:
            exclude_cols = {"subject_id", "sleep_date", "lifelog_date", "date", "timestamp"}
            self.numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        # Keep columns that have at least some non-null values in fitting
        self.numeric_cols = [c for c in self.numeric_cols if X[c].notna().sum() > 0]

        if len(self.numeric_cols) >= self.n_components:
            self.imputer = SimpleImputer(strategy="mean")
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

            imputed_vals = self.imputer.fit_transform(X[self.numeric_cols])
            self.pca.fit(imputed_vals)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.pca is not None and self.imputer is not None:
            imputed_vals = self.imputer.transform(X_copy[self.numeric_cols])
            pca_vals = self.pca.transform(imputed_vals)

            for idx in range(self.n_components):
                X_copy[f"pca_comp_{idx}"] = pca_vals[:, idx]
        return X_copy


class LeakFreeSubjectTargetEncoder(BaseEstimator, TransformerMixin):
    """
    A strictly leak-free target encoder for Subject ID (or user_id)
    utilizing Laplace smoothing fit ONLY on the train target variables.
    """

    def __init__(self, id_col="subject_id", m_smooth=10.0):
        self.id_col = id_col
        self.m_smooth = m_smooth
        self.global_mean_ = 0.5
        self.subject_te_ = None

    def fit(self, X, y):
        # Fit strictly on train target label y
        y_series = pd.Series(y, index=X.index)
        self.global_mean_ = y_series.fillna(0.5).mean()

        non_null_mask = y_series.notna()
        group_sums = y_series[non_null_mask].groupby(X.loc[non_null_mask, self.id_col]).sum()
        group_counts = y_series[non_null_mask].groupby(X.loc[non_null_mask, self.id_col]).count()

        self.subject_te_ = (group_sums + self.m_smooth * self.global_mean_) / (group_counts + self.m_smooth)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["subject_target_encoded"] = X_copy[self.id_col].map(self.subject_te_).fillna(self.global_mean_)
        return X_copy
