# Film Genre Classification — MultinomialNB & Logistic Regression

**Repository for:** "Hyperparameter Optimization Analysis of MultinomialNB and Logistic Regression in Multi-Feature Text-Based Film Genre Classification"  
**Published:** Indonesian Journal on Computing (Vol. 10, Issue 1, 2025)  
**Author / Repo owner:** Shabrio Cahyo Wardoyo — replace with your name and contact details

---

## Project Overview
This repository implements the experiments and code from the thesis that investigates how **Multinomial Naive Bayes (MNB)** and **Logistic Regression (LR)** perform on multi-feature text-based film genre classification. Features include **title**, **description**, and **themes** combined into a single text input. The project emphasizes reproducible preprocessing, TF‑IDF feature extraction, hyperparameter tuning with **GridSearchCV**, and evaluation using standard classification metrics.

**Key result (thesis experiments):** Optimized Logistic Regression achieved an accuracy of **0.847**, while optimized MultinomialNB achieved **0.837**.

---

## Repository Structure (suggested)
```
.
├── data/                     # place dataset CSVs here (e.g., movies.csv, genres.csv, themes.csv)
├── data/processed/           # processed/cleaned data for modeling
├── src/
│   ├── preprocess.py         # cleaning, tokenization, language filter, lemmatization
│   ├── features.py           # TF-IDF pipeline builder
│   ├── train.py              # training, GridSearchCV, save models
│   ├── evaluate.py           # evaluation & report generation (metrics, plots)
│   └── predict.py            # inference script (load model -> predict)
├── notebooks/                # exploratory analysis and figure reproduction
├── models/                   # saved model artifacts (gitignored)
├── requirements.txt
├── README.md                 # this file
└── LICENSE
```

---

## Dataset
The experiments used the **Letterboxd Movies Dataset** (obtained via Kaggle). For reproducibility:
- Download the original CSVs (e.g., `movies.csv`, `genres.csv`, `themes.csv`) and place them in `data/`.
- Follow the preprocessing steps in `src/preprocess.py` to filter, clean, and combine `title + description + themes` into a single text column.
- Note: the original dataset is large and your working dataset will be reduced after cleaning (see thesis for details).

---

## Requirements
Recommended environment:
- Python 3.10+ (tested with Python 3.11)
- Core libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`, `langid` (or `langdetect`), `joblib`, `matplotlib`
- Create a `requirements.txt` with specific versions for reproducibility.

---

## Quickstart
1. Clone the repository:
```bash
git clone https://github.com/shabriocahyo/tugas-akhir-informatika.git
cd film-genre-classifier
```

2. Create virtual environment & install dependencies:
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Place dataset CSVs into `data/`, then preprocess:
```bash
python src/preprocess.py --input-dir data --output-dir data/processed
```

4. Train (example with grid search):
```bash
python src/train.py --data data/processed/combined.csv --model-dir models --grid
```

5. Evaluate:
```bash
python src/evaluate.py --model models/best_lr.joblib --data data/processed/test.csv
```

6. Predict (inference):
```bash
python src/predict.py --model models/best_lr.joblib --input "Title || Description || Themes"
```

---

## Reproducible Configuration (used in thesis)
- **TF‑IDF**: `ngram_range=(1,2)` (unigrams + bigrams), tune `max_features` (e.g., 5000, 10000).
- **MNB Grid**: `alpha` ∈ {0.5, 1.0, 1.5}; tune `tfidf__max_features`.
- **LR Grid**: `C` ∈ {0.1, 1, 10}; tune `tfidf__max_features`.
- **Cross-validation**: `GridSearchCV(cv=5, n_jobs=-1)`.
- **Train/test split**: 80/20.

Adjust grids to match available compute and dataset size.

---

## Evaluation & Metrics
Include standard metrics in `src/evaluate.py`:
- Accuracy, Precision, Recall, F1-score (per-class and macro-averaged)
- Confusion matrix (visualized and saved)
- Training time and grid search duration

Save full classification reports and figures to `results/` or `notebooks/` for reproducibility.

---

## Recommended Scripts / Functions
- `preprocess.py`: language detection/filtering, text cleaning (regex), tokenization, lemmatization/stemming, and feature concatenation.
- `features.py`: build TF‑IDF pipeline with configurable `ngram_range` and `max_features`.
- `train.py`: baseline training, GridSearchCV for MNB and LR, save best estimators and parameters.
- `evaluate.py`: compute metrics, plot confusion matrices, and save reports.
- `predict.py`: inference utility that accepts raw `title|description|themes` input and outputs predicted genre(s).

---

## How to Cite
If you reuse code or results, please cite the original thesis:

> Shabrio Cahyo Wardoyo (2025). *Analisis Optimasi Hyperparameter MultinomialNB dan Logistic Regression dalam Klasifikasi Genre Film Berbasis Teks Multi Fitur*. Universitas Mercu Buana.

---

## License
This repository is recommended to use the **MIT License** for code. Include a `LICENSE` file at the project root.

---

## Contact & Contribution
  - Author: Shabrio Cahyo Wardoyo and Umniy Salamah
  - Email: shabriocahyo
  - GitHub: https://github.com/shabriocahyo

Contributions welcome — add issues or pull requests. Include a `CONTRIBUTING.md` if you expect external collaborators.