# COE292 Project - Social Media Sentiment Analysis

A machine learning project that classifies social media posts as **positive**, **negative**, or **neutral** using KNN, SVM, and a Deep Neural Network (DNN).

---

## Dataset

**File:** `sentiment_analysis.csv`

The dataset contains 494 posts collected from Twitter, Facebook, and Instagram between 2013 and 2023.

| Column | Description |
|---|---|
| `Year`, `Month`, `Day` | Date of the post |
| `Time of Tweet` | Time of day: morning / noon / night |
| `text` | Raw text of the post |
| `sentiment` | Target label: positive / neutral / negative |
| `Platform` | Source platform |

---

## Pipeline

### 1. Data Loading & Preprocessing
- Load CSV and inspect with `data.head()` and `data.info()`
- Missing values in feature columns are imputed with the column mean
- Categorical columns (excluding `text`) are encoded with `LabelEncoder`

### 2. Exploratory Data Analysis
- **Boxplot** of sentiment by year and platform
- **Correlation heatmap** of numeric features vs. sentiment
- **Countplot** of sentiment distribution per platform

### 3. Feature Engineering
- The `text` column is vectorized with **TF-IDF** (`TfidfVectorizer`)
- Remaining numeric features are normalized with `StandardScaler`
- All features are combined horizontally with `np.hstack`
- Residual NaN values are imputed with `SimpleImputer`
- 70/30 train/test split with `random_state=42`

### 4. Models

#### KNN (Euclidean & Manhattan Distance)
- K values swept from 1 to 30; best K selected by test accuracy
- Both distance metrics independently evaluated
- Best K = 1 for both distances (on this dataset)
- Evaluated with accuracy, precision, recall, and 10-fold cross-validation

#### SVM (Hard & Soft Margin)
- **Soft margin:** `SVC(C=1.0, kernel='rbf')`
- **Hard margin:** `SVC(C=100, kernel='rbf')`
- Evaluated with confusion matrix, classification report, and 10-fold cross-validation

#### Deep Neural Network (DNN)
- Architecture: `Dense(16, relu) → Dense(32, relu) → Dense(3, softmax)`
- Optimizer: Adam (lr = 0.0005)
- Loss: Sparse Categorical Crossentropy
- Training: 30 epochs, batch size 30
- Evaluated with confusion matrix, accuracy, precision, and recall

---

## Results Summary

| Model | Test Accuracy |
|---|---|
| KNN (Euclidean, K=1) | ~59% |
| KNN (Manhattan, K=1) | ~59% |
| SVM Soft (C=1) | ~44% |
| SVM Hard (C=100) | ~53% |
| DNN | ~66% |

The DNN achieved the highest test accuracy (~66%), outperforming both KNN and SVM variants on this dataset.

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
keras
```

---

## How to Run

1. Place `sentiment_analysis.csv` in the working directory.
2. Open `COE292_Project.ipynb` in Google Colab or Jupyter.
3. Run all cells sequentially.
