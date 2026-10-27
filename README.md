# ML Regression and Classification Labs

Coursework notebooks (and matching `.py` exports) from a Machine Learning course, covering classic supervised-learning workflows with pandas, scikit-learn, matplotlib and seaborn.

| Notebook | Topic | Data |
|---|---|---|
| `titanic` | Data cleaning, EDA and survival prediction | Titanic CSV (downloaded from a public GitHub URL at run time) |
| `Automobile` | Preprocessing and price modelling | UCI Automobile (`imports-85`), fetched from the UCI repository |
| `linearregression` | Linear regression on student performance and wine quality | `student-por.csv`, `winequality-white.csv` (included) |
| `LogisticRegression` | Binary classification of breast-cancer diagnosis | `wdbc.data` (included, UCI Breast Cancer Wisconsin Diagnostic) |

## Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

Run notebooks from the repository root so the bundled CSV files resolve as relative paths.

## Data sources

- UCI Machine Learning Repository: Student Performance (Portuguese), Wine Quality (white), Breast Cancer Wisconsin (Diagnostic), Automobile.
- Titanic dataset from the public `datasciencedojo/datasets` repository.

These datasets belong to their original providers; see their sources for license terms.
