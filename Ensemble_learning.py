import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works in scripts; in Jupyter/Colab figures are still saved
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42      # fixed seed for reproducibility
TEST_SIZE = 0.20       # 80% training / 20% testing
CV_FOLDS = 5           # folds for stacking, grid search and CV scores
AVERAGE = "weighted"   # averaging used for precision / recall / F1
MALIGNANT = 0          # label 0 = malignant, label 1 = benign in this dataset


# --------------------------------------------------------------------------
# 1. Data loading and inspection
# --------------------------------------------------------------------------
def load_data():
    """Load the Breast Cancer Wisconsin (Diagnostic) dataset as DataFrame/Series."""
    bunch = load_breast_cancer(as_frame=True)
    return bunch.data, bunch.target, [str(n) for n in bunch.target_names]


def describe_dataset(X, y, class_names):
    """Collect and print verified dataset statistics."""
    counts = y.value_counts().sort_index()
    info = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "feature_dtypes": sorted({str(t) for t in X.dtypes}),
        "class_counts": {str(class_names[i]): int(counts[i])
                         for i in counts.index},
        "class_percent": {str(class_names[i]): round(float(100 * counts[i] / len(y)), 2)
                          for i in counts.index},
        "label_mapping": {int(i): str(class_names[i])
                          for i in range(len(class_names))},
    }
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    for key, value in info.items():
        print(f"{key:>16}: {value}")
    return info


# --------------------------------------------------------------------------
# 2. Model construction
# --------------------------------------------------------------------------
def build_base_models():
    """Individual classifiers. Scale-sensitive models use a Pipeline so the
    scaler is fitted on training data only (no data leakage)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        # Trees are scale-invariant, so no scaler is needed
        "Decision Tree": DecisionTreeClassifier(max_depth=5,
                                                random_state=RANDOM_STATE),
        # probability=True is required for soft voting and for stacking
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True,
                        random_state=RANDOM_STATE)),
        ]),
    }


def make_adaboost(**params):
    """AdaBoost with decision stumps.  scikit-learn >= 1.2 uses `estimator`;
    older versions use `base_estimator`, so both are handled."""
    stump = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    try:
        return AdaBoostClassifier(estimator=stump, random_state=RANDOM_STATE,
                                  **params)
    except TypeError:  # scikit-learn < 1.2
        return AdaBoostClassifier(base_estimator=stump,
                                  random_state=RANDOM_STATE, **params)


def tune_adaboost(X_train, y_train, cv):
    """Choose n_estimators and learning_rate using ONLY the training data."""
    grid = GridSearchCV(
        make_adaboost(),
        param_grid={"n_estimators": [50, 100, 200],
                    "learning_rate": [0.1, 0.5, 1.0]},
        scoring="accuracy", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("\nAdaBoost grid search (training data only, "
          f"{CV_FOLDS}-fold CV)")
    print("  best params :", grid.best_params_)
    print(f"  best CV acc : {grid.best_score_:.4f}")
    return grid.best_params_, float(grid.best_score_)


def build_ensemble_models(base_models, adaboost_params, cv):
    """Stacking, AdaBoost and Voting ensembles."""
    estimators = list(base_models.items())
    # Stacking: the meta-classifier is trained on out-of-fold probabilities
    # produced by cross-validation (cv=...), never on predictions of models
    # that already saw the same rows, and never on the test set.
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000,
                                           random_state=RANDOM_STATE),
        cv=cv, stack_method="predict_proba", n_jobs=-1)
    return {
        "Stacking": stacking,
        "AdaBoost": make_adaboost(**adaboost_params),
        "Hard Voting": VotingClassifier(estimators=estimators, voting="hard"),
        "Soft Voting": VotingClassifier(estimators=estimators, voting="soft"),
    }


# --------------------------------------------------------------------------
# 3. Training and evaluation
# --------------------------------------------------------------------------
def evaluate_model(name, model, X_train, y_train, X_test, y_test, cv):
    """Fit on the training set, predict on the shared test set, return metrics."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    y_pred = model.predict(X_test)
    # Rows = actual class, columns = predicted class, label order [0, 1]
    # (0 = malignant, 1 = benign)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # 5-fold CV on the TRAINING data gives a stability estimate per model
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                scoring="accuracy", n_jobs=-1)
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=AVERAGE,
                                     zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=AVERAGE,
                               zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average=AVERAGE, zero_division=0),
        "Malignant Recall": recall_score(y_test, y_pred, pos_label=MALIGNANT),
        "Malignant Precision": precision_score(y_test, y_pred,
                                               pos_label=MALIGNANT,
                                               zero_division=0),
        "CV Acc (mean)": cv_scores.mean(),
        "CV Acc (std)": cv_scores.std(),
        "Fit time (s)": fit_time,
        # Treating "malignant" as the positive class of clinical interest:
        "Malignant missed (FN)": int(cm[0, 1]),   # actual malignant -> predicted benign
        "Benign flagged (FP)": int(cm[1, 0]),     # actual benign -> predicted malignant
    }
    return row, cm, y_pred


def run_experiment(models, X_train, y_train, X_test, y_test, class_names, cv):
    rows, matrices = [], {}
    for name, model in models.items():
        row, cm, y_pred = evaluate_model(name, model, X_train, y_train,
                                         X_test, y_test, cv)
        rows.append(row)
        matrices[name] = cm
        print("\n" + "-" * 70)
        print(f"{name}")
        print("-" * 70)
        print("Confusion matrix (rows = actual, cols = predicted):")
        print(pd.DataFrame(cm, index=[f"Actual {c}" for c in class_names],
                           columns=[f"Pred {c}" for c in class_names]))
        print(classification_report(y_test, y_pred, target_names=class_names,
                                    digits=4))
    return pd.DataFrame(rows), matrices


# --------------------------------------------------------------------------
# 4. Visualisation
# --------------------------------------------------------------------------
def plot_class_distribution(y, class_names, path):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    sns.barplot(x=[class_names[i] for i in counts.index], y=counts.values,
                hue=[class_names[i] for i in counts.index], legend=False,
                palette=["#C0504D", "#4F81BD"], ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 4, str(v), ha="center", fontweight="bold")
    ax.set_title("Class Distribution (Full Dataset)")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Diagnosis")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(matrices, class_names, path):
    n = len(matrices)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, cm) in zip(axes, matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric_comparison(results, path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    long_df = results.melt(id_vars="Model", value_vars=metrics,
                           var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 5.6))
    sns.barplot(data=long_df, x="Model", y="Score", hue="Metric",
                palette="Blues", ax=ax)
    low = max(0.0, long_df["Score"].min() - 0.05)
    ax.set_ylim(low, 1.005)
    ax.set_title("Test-set Performance Comparison "
                 f"({AVERAGE}-averaged precision, recall and F1)")
    ax.set_ylabel(f"Score (y-axis starts at {low:.2f}, not 0)")
    ax.tick_params(axis="x", rotation=20)
    # Legend placed below the axes so it never hides any bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4,
              fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. Self-tests (used for the Test Cases table in the report)
# --------------------------------------------------------------------------
def run_self_tests(X, y, X_test, y_train, y_test, models, results, matrices,
                   ada_params, out):
    """Run nine checks that mirror the test cases in the report. Each check
    returns a measured description and a Pass/Fail status."""
    tests = []
    acc = results.set_index("Model")["Accuracy"]

    def record(tc_id, ok, actual):
        tests.append({"id": tc_id, "status": "Pass" if ok else "Fail",
                      "actual": actual})

    counts = y.value_counts().sort_index()
    record("TC1", X.shape == (569, 30) and set(y.unique()) == {0, 1},
           f"X shape {X.shape}; {len(y)} labels; malignant={counts[0]}, "
           f"benign={counts[1]}")

    tr_share = 100 * (y_train == 0).mean()
    te_share = 100 * (y_test == 0).mean()
    scaled_in_pipeline = all(isinstance(models[m], Pipeline) and
                             "scaler" in models[m].named_steps
                             for m in ("Logistic Regression", "SVM (RBF)"))
    record("TC2", int(X.isna().sum().sum()) == 0 and
           len(y_train) + len(y_test) == len(y) and
           abs(tr_share - te_share) < 1.0 and scaled_in_pipeline,
           f"{int(X.isna().sum().sum())} missing values; train={len(y_train)}, "
           f"test={len(y_test)}; malignant share {tr_share:.2f}% (train) vs "
           f"{te_share:.2f}% (test); scaler inside Pipeline")

    base_names = ["Logistic Regression", "Decision Tree", "SVM (RBF)"]
    record("TC3", all(acc[m] > 0.90 for m in base_names),
           "; ".join(f"{m} acc={acc[m]:.4f}" for m in base_names))

    stack = models["Stacking"]
    stack_ok = (len(stack.estimators_) == 3 and
                isinstance(stack.final_estimator_, LogisticRegression) and
                getattr(stack.cv, "n_splits", None) == CV_FOLDS and
                acc["Stacking"] > 0.90)
    record("TC4", stack_ok,
           f"{len(stack.estimators_)} base models + "
           f"{type(stack.final_estimator_).__name__} meta-classifier; "
           f"{stack.cv.n_splits}-fold out-of-fold predictions; "
           f"acc={acc['Stacking']:.4f}")

    ada = models["AdaBoost"]
    n_fit = len(ada.estimators_)
    record("TC5", 0 < n_fit <= ada_params["n_estimators"] and
           acc["AdaBoost"] > 0.90,
           f"{n_fit} weak learners fitted (max_depth=1 stumps); "
           f"params={ada_params}; acc={acc['AdaBoost']:.4f}")

    soft_proba = models["Soft Voting"].predict_proba(X_test)
    record("TC6", acc["Hard Voting"] > 0.90 and acc["Soft Voting"] > 0.90 and
           np.allclose(soft_proba.sum(axis=1), 1.0),
           f"Hard voting acc={acc['Hard Voting']:.4f}; Soft voting "
           f"acc={acc['Soft Voting']:.4f}; probability rows sum to 1")

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score"]
    in_range = bool(((results[metric_cols] >= 0) &
                     (results[metric_cols] <= 1)).all().all())
    consistent = all(np.isclose(np.trace(matrices[m]) / matrices[m].sum(),
                                acc[m]) for m in matrices)
    record("TC7", in_range and consistent,
           "All four metrics lie in [0, 1]; accuracy recomputed from each "
           "confusion matrix equals accuracy_score")

    record("TC8", all(cm.shape == (2, 2) and cm.sum() == len(y_test)
                      for cm in matrices.values()),
           f"{len(matrices)} matrices of shape (2, 2); each sums to "
           f"{len(y_test)} test samples")

    files = ["results_table.csv", "class_distribution.png",
             "confusion_matrices.png", "metrics_comparison.png"]
    present = all((out / f).exists() and (out / f).stat().st_size > 0
                  for f in files)
    record("TC9", present and len(results) == 7,
           f"Comparison table with {len(results)} models and 3 figures saved")

    print("\n" + "=" * 70)
    print("SELF-TESTS")
    print("=" * 70)
    for t in tests:
        print(f"{t['id']}: {t['status']} - {t['actual']}")
    return tests


# --------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------
def main(out_dir="outputs_practical5"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Python {platform.python_version()} | scikit-learn "
          f"{sklearn.__version__} | pandas {pd.__version__} | "
          f"numpy {np.__version__} | seaborn {sns.__version__}")

    X, y, class_names = load_data()
    info = describe_dataset(X, y, class_names)

    # Stratified split keeps the class ratio equal in both subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    split_info = {
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "train_class_counts": {str(class_names[i]): int(c) for i, c in
                               y_train.value_counts().sort_index().items()},
        "test_class_counts": {str(class_names[i]): int(c) for i, c in
                              y_test.value_counts().sort_index().items()},
    }
    print("\nTrain/test split:", split_info)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    base_models = build_base_models()
    ada_params, ada_cv = tune_adaboost(X_train, y_train, cv)
    ensemble_models = build_ensemble_models(base_models, ada_params, cv)
    all_models = {**build_base_models(), **ensemble_models}

    results, matrices = run_experiment(all_models, X_train, y_train,
                                       X_test, y_test, class_names, cv)

    # Comparison table
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (held-out test set)")
    print("=" * 70)
    print(results.round(4).to_string(index=False))
    results.to_csv(out / "results_table.csv", index=False)

    top = results["F1-score"].round(4).max()
    leaders = results.loc[results["F1-score"].round(4) == top, "Model"].tolist()
    print(f"\nHighest weighted F1 on the test set ({top:.4f}): "
          + ", ".join(leaders))
    if len(leaders) > 1:
        print("Several models tie, and with only "
              f"{len(y_test)} test samples one error changes accuracy by "
              f"{100 / len(y_test):.2f} percentage points.")

    plot_class_distribution(y, class_names, out / "class_distribution.png")
    plot_confusion_matrices(matrices, class_names, out / "confusion_matrices.png")
    plot_metric_comparison(results, out / "metrics_comparison.png")

    tests = run_self_tests(X, y, X_test, y_train, y_test, all_models, results,
                           matrices, ada_params, out)

    payload = {
        "versions": {"python": platform.python_version(),
                     "sklearn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "seaborn": sns.__version__,
                     "matplotlib": matplotlib.__version__},
        "self_tests": tests,
        "dataset": info, "split": split_info,
        "adaboost_params": ada_params, "adaboost_cv_accuracy": ada_cv,
        "results": results.to_dict(orient="records"),
        "confusion_matrices": {k: v.tolist() for k, v in matrices.items()},
        "settings": {"random_state": RANDOM_STATE, "test_size": TEST_SIZE,
                     "cv_folds": CV_FOLDS, "average": AVERAGE},
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved tables and figures to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs_practical5")
    # parse_known_args keeps the script usable inside Jupyter / Colab
    args, _ = parser.parse_known_args()
    main(args.out)