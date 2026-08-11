# Examples & Results

Interactive Jupyter notebooks in [`static/`](https://github.com/qiancapital/snapboost/tree/main/static) walk through classification, regression, and hyperparameter exploration. Each notebook trains SnapBoost and compares it against **XGBoost** and **LightGBM** on the same splits.

| Notebook | Dataset | SnapBoost | XGBoost | LightGBM |
|----------|---------|-----------|---------|----------|
| [Classification.ipynb](https://github.com/qiancapital/snapboost/blob/main/static/Classification.ipynb) | Breast Cancer Wisconsin | 97.2% accuracy | 95.8% | 96.5% |
| [Regression.ipynb](https://github.com/qiancapital/snapboost/blob/main/static/Regression.ipynb) | Diabetes | R² 0.44, RMSE 55.7 | R² 0.38, RMSE 58.4 | R² 0.40, RMSE 57.7 |
| [Parameter_Exploration.ipynb](https://github.com/qiancapital/snapboost/blob/main/static/Parameter_Exploration.ipynb) | Synthetic (piecewise + smooth) | R² 0.986, RMSE 0.170 | R² 0.986, RMSE 0.174 | R² 0.987, RMSE 0.167 |

Run the notebooks locally:

```bash
pip install -r requirements.txt xgboost lightgbm
jupyter notebook static/
```

## Classification

On the Breast Cancer dataset (250 boosting rounds), SnapBoost achieves the highest test accuracy and fewest misclassifications among the three boosters.

```{image} _static/figures/classification_comparison.png
:alt: Test accuracy and error count vs XGBoost and LightGBM
:width: 720px
```

Confusion matrix for SnapBoost on the held-out test set:

```{image} _static/figures/classification_confusion_matrix.png
:alt: SnapBoost classification confusion matrix
:width: 480px
```

## Regression

On the Diabetes dataset (100 boosting rounds), SnapBoost improves R² and RMSE over tree-only baselines.

```{image} _static/figures/regression_comparison.png
:alt: R², RMSE, and MAE comparison on Diabetes dataset
:width: 720px
```

Predicted vs. actual disease progression:

```{image} _static/figures/regression_predicted_vs_actual.png
:alt: Predicted vs actual scatter plot
:width: 560px
```

SnapBoost fitted curve along BMI (other features held at training medians):

```{image} _static/figures/regression_bmi_fit.png
:alt: BMI vs target with SnapBoost fit
:width: 560px
```

## Parameter exploration

On a synthetic dataset mixing piecewise-linear and sinusoidal structure, a mixed ensemble (`p_tree=0.8`) outperforms trees-only (`p_tree=1.0`, RMSE 0.174) and ridge-only (`p_tree=0.0`, RMSE 0.366):

```{image} _static/figures/parameter_exploration_predictions.png
:alt: Learned functions along one axis for different p_tree values
:width: 720px
```
