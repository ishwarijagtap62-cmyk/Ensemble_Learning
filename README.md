# Ensemble Learning and Performance Evaluation

**TY B.Tech. – Artificial Intelligence and Machine Learning | Practical Assignment 5**

Implementation and comparison of ensemble learning techniques (**Stacking**, **AdaBoost**, **Hard Voting** and **Soft Voting**) against individual classifiers (**Logistic Regression**, **Decision Tree**, **SVM**) on the Breast Cancer Wisconsin (Diagnostic) dataset.

Author: Ishwari Vishwajit Jagtap
Faculty: Dr. Khushal Khairnar

---

## Problem Statement

To implement and evaluate ensemble learning techniques, including Stacking, AdaBoost, and Voting Classifiers, and compare their performance with individual base machine learning classifiers using appropriate classification metrics.

## Dataset

| Item | Details |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17); a copy is bundled with scikit-learn (`load_breast_cancer`) |
| Size | 569 samples, 30 numeric features |
| Target | 0 = malignant (212), 1 = benign (357) |
| Missing values | None |
| Split | 80% train / 20% test, stratified, `random_state=42` (455 / 114 samples) |

## Models

| Model | Configuration |
|---|---|
| Logistic Regression | `StandardScaler` + `LogisticRegression` (Pipeline) |
| Decision Tree | `max_depth=5` |
| SVM (RBF) | `StandardScaler` + `SVC(probability=True)` (Pipeline) |
| Stacking | Base: LR, DT, SVM; meta-classifier: Logistic Regression; 5-fold stratified CV (out-of-fold predictions) |
| AdaBoost | Decision stumps; `n_estimators` and `learning_rate` chosen by 5-fold grid search on training data only |
| Hard Voting | Majority vote of LR, DT, SVM |
| Soft Voting | Averaged probabilities of LR, DT, SVM |

All models are trained on the training set only and evaluated on the same held-out test set. Scaling lives inside Pipelines, so no information leaks from the test data.

## Results (test set, 114 samples)

Precision, recall and F1 are weighted averages over both classes.

| Model | Accuracy | Precision | Recall | F1-score | Malignant missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9825 | 0.9825 | 0.9825 | 1 | 1 |
| Decision Tree | 0.9211 | 0.9234 | 0.9211 | 0.9216 | 3 | 6 |
| SVM (RBF) | 0.9825 | 0.9825 | 0.9825 | 0.9825 | 1 | 1 |
| Stacking | 0.9737 | 0.9740 | 0.9737 | 0.9737 | 1 | 2 |
| AdaBoost | 0.9561 | 0.9569 | 0.9561 | 0.9558 | 4 | 1 |
| Hard Voting | 0.9825 | 0.9825 | 0.9825 | 0.9825 | 1 | 1 |
| Soft Voting | 0.9737 | 0.9737 | 0.9737 | 0.9736 | 2 | 1 |

**Key findings**

- Logistic Regression, SVM and Hard Voting tie for the best test result.
- Ensembles clearly improved on the weak Decision Tree but did **not** outperform the strongest individual classifiers on this dataset.
- With only 114 test samples, one extra error changes accuracy by 0.88 percentage points, so small gaps between the top models are not conclusive.

Full tables, confusion matrices, charts and analysis are in the report.

## Repository Contents

```
├── ensemble_learning_practical5.py   # complete source code
├── Practical_Assignment_5_Ensemble_Learning_and_Performance_Evaluation.docx   # full report
└── README.md
```

Running the script also creates an output folder with `results_table.csv`, `results.json`, `class_distribution.png`, `confusion_matrices.png` and `metrics_comparison.png`.

## Requirements

Results were produced with:

- Python 3.12.3
- scikit-learn 1.8.0
- pandas 3.0.2
- NumPy 2.4.4
- Matplotlib 3.10.8
- Seaborn 0.13.2

Install the libraries:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

The AdaBoost helper handles both the newer `estimator` argument and the older `base_estimator` argument, so the script also runs on older scikit-learn versions.

## How to Run

```bash
python ensemble_learning_practical5.py
# optional: choose an output folder
python ensemble_learning_practical5.py --out my_results
```

The script also works in Jupyter Notebook or Google Colab. It prints the dataset summary, classification reports, confusion matrices, the comparison table and nine self-tests that check each stage of the pipeline.

## Reproducibility

A fixed random seed (`42`) is used for the split, the models and the cross-validation. Metrics were identical across repeated runs; training times vary slightly between machines.

## Limitations and Responsible Use

- Small test set and a single train/test split; repeated or nested cross-validation and significance tests would give stronger evidence.
- Single data source with no demographic attributes, so fairness across patient groups could not be evaluated.
- This project is for **educational purposes only** and must not be used for real medical diagnosis.

## References

1. UCI Machine Learning Repository, *Breast Cancer Wisconsin (Diagnostic)*. https://archive.ics.uci.edu/dataset/17
2. scikit-learn documentation, *Ensembles: Gradient boosting, random forests, bagging, voting, stacking*. https://scikit-learn.org/1.8/modules/ensemble.html
3. Y. Freund and R. E. Schapire, "A decision-theoretic generalization of on-line learning and an application to boosting", *Journal of Computer and System Sciences*, 55(1), 119–139, 1997. https://doi.org/10.1006/jcss.1997.1504
4. D. H. Wolpert, "Stacked generalization", *Neural Networks*, 5(2), 241–259, 1992.
