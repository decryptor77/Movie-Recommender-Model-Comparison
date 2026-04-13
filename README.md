# Movie Recommender Model Comparison

A recommendation systems project completed during my **MSc in Data Science & Machine Learning at Reichman University**, in the frame of a **Recommendation Systems** course.  
The project implements, optimizes, tunes, and compares several recommender models on the **MovieLens 100K** dataset, with both quantitative evaluation and qualitative model explainability analysis.

> Final project outcome: **100/100**  
> Feedback: **"Perfect submission. Excellent implementation and analysis".**

---

## Overview

This project builds and evaluates four recommendation models from scratch:

- **Popularity model**
- **Bias-only model**
- **Matrix Factorization with Gradient Descent (GD)**
- **Matrix Factorization with Alternating Least Squares (ALS)**

The work emphasizes:

- correct handling of sparse explicit-feedback data
- vectorized NumPy/Pandas implementations
- hyperparameter tuning for ALS and GD
- comparison across prediction and ranking metrics
- qualitative item-to-item recommendation analysis via model-aware similarity

---

## Dataset

The project uses the **MovieLens 100K** dataset:

- $100{,}000$ explicit ratings
- $943$ users
- $1{,}682$ movies
- rating scale: $1$ to $5$

The rating matrix is constructed with missing ratings filled as $0$, while explicitly treating those zeros as **missing values**, not true ratings.

---

## Implemented Models

### 1. Popularity
Predicts each item using its average observed rating.

### 2. Bias Model
Models ratings using global, user, and item effects:

$r_{ui} \approx \mu + b_u + b_i$

### 3. Gradient Descent Matrix Factorization
Learns latent user and item factors:

$r_{ui} \approx U_u \cdot V_i$

using regularized batch gradient descent.

### 4. Alternating Least Squares
Learns latent factors by alternating between closed-form regularized least-squares updates for users and items.

---

## Evaluation

The models are evaluated using both regression and ranking metrics:

- **RMSE**
- **MRR@$K$**
- **MAP@$K$**
- **nDCG@$K$**

This allows comparison from two perspectives:

- how accurately the models predict ratings
- how well they rank useful recommendations

---

## Main Technical Contributions

### Vectorized implementation
A major part of the project focused on improving implementation efficiency using **NumPy vectorization**.

This includes:

- vectorized `predict_all()`
- vectorized RMSE computation
- sparse-aware gradient updates for GD
- optimized training and evaluation flow
- reduced dependence on nested Python loops

These changes substantially improved runtime and made repeated experiments and grid search practical.

### Hyperparameter tuning
Both **ALS** and **GD** were tuned through grid search.

Key findings from tuning:

- **ALS** benefited strongly from increased regularization, with overfitting clearly visible at lower regularization values.
- The best ALS configuration found in the extended search was:

  - $n\_factors = 3$
  - $reg = 2$
  - $n\_iter = 10$

- The best GD configuration found was:

  - $n\_factors = 3$
  - $reg = 0.001$
  - $n\_iter = 150$
  - $learning\_rate = 0.001$

### Model explainability
For qualitative comparison, item-to-item similarity was computed using a **model-aware approach**:

- for **Popularity** and **Bias**: cosine similarity over predicted rating profiles
- for **GD** and **ALS**: cosine similarity over learned item embeddings

This was done to ensure that similarity reflects the representation naturally produced by each model.

---

## Key Findings

- Simple baselines such as **Popularity** and **Bias** are fast and useful as reference points, but they do not capture meaningful item structure.
- **Matrix Factorization** models provide stronger personalization and better recommendation quality.
- **ALS** produced the strongest overall qualitative recommendation behavior.
- Compared with GD, ALS yielded **smoother and more stable latent item embeddings**, consistent with its regularized least-squares optimization.
- Different models perform better under different criteria: some are attractive for speed and simplicity, while others are preferable for personalized recommendation quality.

---

## Qualitative Recommendation Analysis

The project also compares item-to-item recommendations for selected movies, including:

- *GoldenEye (1995)*
- *Seven (Se7en) (1995)*
- *The Usual Suspects (1995)*

The qualitative analysis shows clear differences between models:

- **Popularity** tends to return movies driven by broad popularity effects rather than semantic similarity
- **Bias** can surface highly rated or well-known movies, but remains inconsistent
- **GD** captures some latent structure but often produces noisier neighborhoods
- **ALS** generally yields more coherent and interpretable recommendations

---

## Repository Contents

```text
.
├── movie_recommender_model_comparison.ipynb  
└── README.md                               
```

---

## Notes

- Results for GD and ALS may vary slightly across runs if a random seed is not fixed.
- Some display behavior in the qualitative comparison depends on helper functions provided in the assignment notebook.
- The project prioritizes correctness, analysis, and efficient implementation in an academic setting.

---

## Author

**Noor Nashef**  
MSc Data Science & Machine Learning student, Reichman University
BSc in Information Systems Engineering specialized in Machine Learning, Technion

