"""Leakage-safe LightGBM training core for jaw cystic lesion classification.

Scope: pooled out-of-fold development-cohort evaluation. An independent test
set must remain outside this function and must not be used for model selection.
"""

from __future__ import annotations

import copy
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ["A", "C", "K"]
CLASS_LABELS = np.arange(len(CLASS_NAMES))


def _as_numeric_frame(X: Any) -> pd.DataFrame:
    """Return a numeric, finite-or-missing DataFrame without changing columns."""
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if not frame.columns.is_unique:
        raise ValueError("Feature names must be unique.")
    frame = frame.apply(pd.to_numeric, errors="raise")
    return frame.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


class StabilityFeatureSelector:
    """Fold-local median imputation, filtering, MI ranking, and L1 stability.

    The operations follow the manuscript: variance threshold 0.01 after
    standardisation, Spearman |r| > 0.9 pruning, top-k mutual-information
    candidates, and bootstrapped L1-logistic selection at an 80% threshold.
    """

    def __init__(
        self,
        varnum: int = 25,
        variance_threshold: float = 0.01,
        corr_threshold: float = 0.9,
        stability_threshold: float = 0.8,
        n_bootstrap: int = 50,
        l1_c: float = 1.0,
        random_state: int = 42,
        max_iter: int = 5000,
    ):
        self.varnum = varnum
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.stability_threshold = stability_threshold
        self.n_bootstrap = n_bootstrap
        self.l1_c = l1_c
        self.random_state = random_state
        self.max_iter = max_iter

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose constructor parameters for scikit-learn cloning/grid search."""
        return {
            "varnum": self.varnum,
            "variance_threshold": self.variance_threshold,
            "corr_threshold": self.corr_threshold,
            "stability_threshold": self.stability_threshold,
            "n_bootstrap": self.n_bootstrap,
            "l1_c": self.l1_c,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
        }

    def set_params(self, **params: Any) -> "StabilityFeatureSelector":
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)
        return self

    def _validate_hyperparameters(self) -> None:
        if not isinstance(self.varnum, (int, np.integer)) or self.varnum < 1:
            raise ValueError("varnum must be a positive integer.")
        if not isinstance(self.n_bootstrap, (int, np.integer)) or self.n_bootstrap < 2:
            raise ValueError("n_bootstrap must be an integer >= 2.")
        if not 0 < self.stability_threshold <= 1:
            raise ValueError("stability_threshold must be in (0, 1].")
        if self.l1_c <= 0:
            raise ValueError("l1_c must be positive.")

    def fit(self, X: Any, y: Any) -> "StabilityFeatureSelector":
        self._validate_hyperparameters()
        frame = _as_numeric_frame(X)
        labels = np.asarray(y, dtype=int).reshape(-1)
        if len(frame) != len(labels):
            raise ValueError("X and y must contain the same number of samples.")

        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.impute_values_ = frame.median(axis=0)
        all_missing = frame.columns[self.impute_values_.isna()]
        if len(all_missing):
            raise ValueError(
                f"Columns with no observed training values: {all_missing.tolist()}"
            )

        imputed = frame.fillna(self.impute_values_)
        self.scaler_ = StandardScaler()
        scaled = pd.DataFrame(
            self.scaler_.fit_transform(imputed),
            columns=frame.columns,
            index=frame.index,
        )

        self.variance_filter_ = VarianceThreshold(
            threshold=self.variance_threshold
        )
        self.variance_filter_.fit(scaled)
        variance_features = scaled.columns[self.variance_filter_.get_support()]
        if len(variance_features) == 0:
            raise ValueError("No features remain after variance filtering.")
        variance_frame = scaled.loc[:, variance_features]

        correlation = variance_frame.corr(method="spearman").abs()
        upper = correlation.where(
            np.triu(np.ones(correlation.shape), k=1).astype(bool)
        )
        self.correlation_dropped_ = [
            column for column in upper.columns if any(upper[column] > self.corr_threshold)
        ]
        uncorrelated = variance_frame.drop(columns=self.correlation_dropped_)
        if uncorrelated.shape[1] == 0:
            raise ValueError("No features remain after correlation pruning.")

        # Imported lazily so cloning and grid-search workers stay lightweight.
        from sklearn.feature_selection import mutual_info_classif

        mi = mutual_info_classif(
            uncorrelated,
            labels,
            random_state=self.random_state,
        )
        mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        self.mi_scores_ = pd.Series(mi, index=uncorrelated.columns, name="mi_score")
        k = min(self.varnum, uncorrelated.shape[1])
        candidate_index = np.argsort(-mi, kind="stable")[:k]
        self.candidate_features_ = uncorrelated.columns[candidate_index].tolist()
        candidates = uncorrelated.loc[:, self.candidate_features_]

        rng = np.random.default_rng(self.random_state)
        selected_counts = np.zeros(k, dtype=int)
        class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        if any(len(indices) == 0 for indices in class_indices):
            raise ValueError("Each bootstrap replicate must contain every class.")

        # Stratified bootstrap preserves the development-sample class counts.
        for _ in range(self.n_bootstrap):
            bootstrap_parts = [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
            bootstrap_index = np.concatenate(bootstrap_parts)
            rng.shuffle(bootstrap_index)
            estimator = LogisticRegression(
                penalty="l1",
                solver="saga",
                C=self.l1_c,
                class_weight="balanced",
                max_iter=self.max_iter,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            estimator.fit(candidates.iloc[bootstrap_index], labels[bootstrap_index])
            selected_counts += (np.abs(estimator.coef_) > 1e-8).any(axis=0)

        frequencies = selected_counts / float(self.n_bootstrap)
        self.selection_frequencies_ = pd.Series(
            frequencies,
            index=self.candidate_features_,
            name="selection_frequency",
        ).sort_values(ascending=False)
        self.selected_features_ = self.selection_frequencies_.index[
            self.selection_frequencies_ >= self.stability_threshold
        ].tolist()
        if not self.selected_features_:
            raise ValueError(
                "Stability selection retained no features. Review the prespecified "
                "threshold/model settings rather than selecting features on test data."
            )
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("The selector must be fitted before transform().")
        frame = _as_numeric_frame(X)
        if list(frame.columns) != self.feature_names_in_.tolist():
            raise ValueError("Transform columns differ from the fitted training columns.")
        imputed = frame.fillna(self.impute_values_)
        scaled = pd.DataFrame(
            self.scaler_.transform(imputed),
            columns=frame.columns,
            index=frame.index,
        )
        return scaled.loc[:, self.selected_features_]

    def fit_transform(self, X: Any, y: Any, **fit_params: Any) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.selected_features_, dtype=object)


def _build_pipeline(
    random_state: int,
    varnum: int,
    n_bootstrap: int,
    stability_threshold: float,
) -> Any:
    """Build a pipeline whose every fitted step is restricted to training data."""
    from sklearn.pipeline import Pipeline

    selector = StabilityFeatureSelector(
        varnum=varnum,
        n_bootstrap=n_bootstrap,
        stability_threshold=stability_threshold,
        random_state=random_state,
    )
    classifier = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_NAMES),
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        subsample_freq=1,  # Required for subsample to take effect in LightGBM.
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )
    return Pipeline([("feature_selection", selector), ("classifier", classifier)])


def _normalise_shap_shape(
    raw_values: Any,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Return SHAP values as (samples, features, classes) across SHAP versions."""
    if isinstance(raw_values, list):
        values = np.stack(raw_values, axis=-1)
    else:
        values = np.asarray(raw_values)
    if values.ndim == 2:
        values = values[:, :, np.newaxis]
    if values.ndim != 3:
        raise ValueError(f"Unrecognised SHAP dimensionality: {values.ndim}")
    if values.shape == (n_samples, n_features, len(CLASS_NAMES)):
        return values
    if values.shape == (n_samples, len(CLASS_NAMES), n_features):
        return np.transpose(values, (0, 2, 1))
    raise ValueError(f"Unrecognised SHAP shape: {values.shape}")


def _validate_training_data(
    X: Any,
    y: Any,
    n_folds: int,
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.Series]:
    frame = _as_numeric_frame(X)
    labels_array = np.asarray(y).reshape(-1)
    if len(frame) != len(labels_array):
        raise ValueError("X and y must contain the same number of samples.")
    if not np.all(np.isin(labels_array, CLASS_LABELS)):
        raise ValueError("y must use contiguous class labels: 0 (A), 1 (C), 2 (K).")
    labels = pd.Series(labels_array, dtype=int, name="label")
    class_counts = labels.value_counts()
    minimum_required = max(n_folds, inner_folds)
    if class_counts.min() < minimum_required:
        raise ValueError(
            f"Each class needs at least {minimum_required} samples for the requested "
            f"cross-validation; observed counts: {class_counts.to_dict()}"
        )
    return frame, labels


def model_train_and_evaluate(
    X: Any,
    y: Any,
    random_state_1: int = 42,
    varnum_1: int = 25,
    shap_save_path: str = "shap_beeswarm_all_validation_samples.png",
    *,
    n_folds: int = 5,
    inner_folds: int = 3,
    n_bootstrap: int = 50,
    stability_threshold: float = 0.8,
    param_grid: dict[str, list[Any]] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run nested, leakage-safe LightGBM development cross-validation.

    Outer folds provide out-of-fold predictions and SHAP values. Inner folds
    optimise LightGBM hyperparameters using only the corresponding outer
    training data. No outer validation fold is used for early stopping.

    This function does not perform independent-test evaluation. Split off the
    test cohort before calling it, and never include that cohort in X or y.

    Additional result fields record fold-wise selected features, best inner-CV
    parameters, and original sample indices. Existing downstream plotting code
    can continue to use the original SHAP fields.
    """
    X, y = _validate_training_data(X, y, n_folds, inner_folds)
    outer_cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state_1,
    )
    if param_grid is None:
        param_grid = {
            "classifier__learning_rate": [0.05, 0.1, 0.2],
            "classifier__num_leaves": [15, 31, 63],
        }

    fold_acc_list: list[float] = []
    fold_auc_list: list[float] = []
    fold_class_auc_list: list[list[float]] = []
    fold_best_params: list[dict[str, Any]] = []
    fold_selected_features: list[list[str]] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_sample_indices: list[np.ndarray] = []
    all_y_pred_prob: list[np.ndarray] = []
    fold_shap_values: list[np.ndarray] = []
    fold_X_val: list[pd.DataFrame] = []
    fold_pred_classes: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"\nOuter fold {fold}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        inner_cv = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=random_state_1,
        )
        search = GridSearchCV(
            estimator=_build_pipeline(
                random_state=random_state_1,
                varnum=varnum_1,
                n_bootstrap=n_bootstrap,
                stability_threshold=stability_threshold,
            ),
            param_grid=param_grid,
            scoring="roc_auc_ovr",
            cv=inner_cv,
            refit=True,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        selector = best_pipeline.named_steps["feature_selection"]
        classifier = best_pipeline.named_steps["classifier"]
        X_val_selected = selector.transform(X_val)

        y_pred_prob = best_pipeline.predict_proba(X_val)
        if not np.array_equal(classifier.classes_, CLASS_LABELS):
            raise ValueError(f"Unexpected class order: {classifier.classes_}")
        y_pred = np.argmax(y_pred_prob, axis=1)

        acc = accuracy_score(y_val, y_pred)
        auc_macro = roc_auc_score(
            y_val,
            y_pred_prob,
            labels=CLASS_LABELS,
            multi_class="ovr",
            average="macro",
        )
        class_aucs = []
        for class_index in CLASS_LABELS:
            y_true_bin = (np.asarray(y_val) == class_index).astype(int)
            class_aucs.append(roc_auc_score(y_true_bin, y_pred_prob[:, class_index]))

        fold_acc_list.append(acc)
        fold_auc_list.append(auc_macro)
        fold_class_auc_list.append(class_aucs)
        fold_best_params.append(copy.deepcopy(search.best_params_))
        fold_selected_features.append(selector.selected_features_.copy())
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_sample_indices.append(val_idx.copy())
        all_y_pred_prob.append(y_pred_prob)

        explainer = shap.TreeExplainer(classifier)
        shap_values_fold = _normalise_shap_shape(
            explainer.shap_values(X_val_selected, check_additivity=False),
            n_samples=len(X_val_selected),
            n_features=X_val_selected.shape[1],
        )
        fold_shap_values.append(shap_values_fold)
        fold_X_val.append(X_val_selected.copy())
        fold_pred_classes.append(y_pred)

        print(
            f"accuracy={acc:.4f}; macro-OvR-AUC={auc_macro:.4f}; "
            f"features={len(selector.selected_features_)}; "
            f"best_params={search.best_params_}"
        )
        print("Confusion matrix:\n", confusion_matrix(y_val, y_pred, labels=CLASS_LABELS))

    probabilities = np.concatenate(all_y_pred_prob, axis=0)
    # Fold-local selection can produce different feature sets. Align SHAP values
    # to their union: an unselected feature has zero attribution in that fold,
    # while its display value remains missing rather than being fabricated.
    feature_names = list(
        dict.fromkeys(
            name for fold_frame in fold_X_val for name in fold_frame.columns
        )
    )
    aligned_shap = []
    aligned_frames = []
    for fold_values, fold_frame in zip(fold_shap_values, fold_X_val):
        aligned_values = np.zeros(
            (len(fold_frame), len(feature_names), len(CLASS_NAMES)),
            dtype=float,
        )
        aligned_frame = pd.DataFrame(
            np.nan,
            index=fold_frame.index,
            columns=feature_names,
        )
        for local_index, name in enumerate(fold_frame.columns):
            global_index = feature_names.index(name)
            aligned_values[:, global_index, :] = fold_values[:, local_index, :]
            aligned_frame[name] = fold_frame[name].to_numpy()
        aligned_shap.append(aligned_values)
        aligned_frames.append(aligned_frame)
    shap_values = np.concatenate(aligned_shap, axis=0)
    X_val_shap = pd.concat(aligned_frames, axis=0, ignore_index=True)
    predicted_classes = np.concatenate(fold_pred_classes, axis=0)

    shap_by_class = np.mean(np.abs(shap_values), axis=0)
    shap_overall = np.mean(shap_by_class, axis=1)
    shap_importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": shap_overall}
    )
    for index, class_name in enumerate(CLASS_NAMES):
        shap_importance[f"mean_abs_shap_{class_name}"] = shap_by_class[:, index]
    shap_importance = shap_importance.sort_values(
        by="mean_abs_shap", ascending=False
    ).reset_index(drop=True)

    global_auc_macro = roc_auc_score(
        all_y_true,
        probabilities,
        labels=CLASS_LABELS,
        multi_class="ovr",
        average="macro",
    )
    report = classification_report(
        all_y_true,
        all_y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    class_auc_array = np.asarray(fold_class_auc_list)
    mean_class_aucs = np.mean(class_auc_array, axis=0)
    std_class_aucs = np.std(class_auc_array, axis=0)

    result_row: dict[str, Any] = {
        "Accuracy_overall": report["accuracy"],
        "std_acc": np.std(fold_acc_list),
        "mean_auc": np.mean(fold_auc_list),
        "std_auc": np.std(fold_auc_list),
        "AUC_macro_OvR": global_auc_macro,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }
    for index, class_name in enumerate(CLASS_NAMES):
        result_row[f"AUC_OvR_{class_name}"] = mean_class_aucs[index]
        result_row[f"AUC_OvR_std_{class_name}"] = std_class_aucs[index]
        result_row[f"precision_{class_name}"] = report[class_name]["precision"]
        result_row[f"recall_{class_name}"] = report[class_name]["recall"]
        result_row[f"f1_{class_name}"] = report[class_name]["f1-score"]

    summary = pd.DataFrame([result_row])
    summary["all_y_true"] = [all_y_true]
    summary["all_y_pred"] = [all_y_pred]
    summary["all_y_pred_prob"] = [probabilities]
    summary["all_sample_indices"] = [np.concatenate(all_sample_indices)]
    summary["shap_feature_names"] = [shap_importance["feature"].tolist()]
    summary["shap_mean_abs_importance"] = [
        shap_importance["mean_abs_shap"].tolist()
    ]
    summary["shap_importance_by_class"] = [
        {
            class_name: shap_importance[f"mean_abs_shap_{class_name}"].tolist()
            for class_name in CLASS_NAMES
        }
    ]
    summary["shap_importance_table"] = [
        shap_importance.to_dict(orient="records")
    ]
    summary["shap_beeswarm_path"] = [shap_save_path]
    summary["all_shap_values"] = [shap_values]
    summary["all_X_val_shap"] = [X_val_shap]
    summary["all_pred_classes_for_shap"] = [predicted_classes]
    summary["fold_best_params"] = [fold_best_params]
    summary["fold_selected_features"] = [fold_selected_features]
    summary["random_state_1"] = random_state_1
    summary["varnum_1"] = varnum_1
    summary["n_bootstrap"] = n_bootstrap
    summary["stability_threshold"] = stability_threshold

    print("\nPooled out-of-fold summary (development data only):")
    print(summary.drop(columns=["all_shap_values", "all_X_val_shap"]))
    return summary


__all__ = ["StabilityFeatureSelector", "model_train_and_evaluate"]
