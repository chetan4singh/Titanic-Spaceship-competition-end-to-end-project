# Titanic-Spaceship-competition-end-to-end-project# 🚀 Spaceship Titanic — Kaggle Competition (End-to-End ML Project)

An end-to-end machine learning project solving Kaggle's [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition — a binary classification problem where the goal is to predict whether a passenger was **transported to an alternate dimension** after the spaceship *Titanic* collided with a spacetime anomaly.

## 📖 Problem Statement

It's the year 2912. The spaceship *Titanic*, carrying 13,000 passengers to new planets, collided with a spacetime anomaly near Alpha Centauri. While the ship survived, **~half the passengers vanished into another dimension**. Using passenger records recovered from the ship's damaged computer system, the task is to predict which passengers were transported — helping rescue crews retrieve them.

## 🗂️ Dataset

The dataset is provided by Kaggle and includes `train.csv` and `test.csv` with fields such as:

| Feature | Description |
|---|---|
| `PassengerId` | Unique ID (`gggg_pp` — group and position within group) |
| `HomePlanet` | Planet the passenger departed from |
| `CryoSleep` | Whether the passenger was in suspended animation |
| `Cabin` | Cabin number (`deck/num/side`) |
| `Destination` | Planet the passenger was traveling to |
| `Age` | Passenger's age |
| `VIP` | Whether the passenger paid for VIP service |
| `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` | Amount billed at ship amenities |
| `Name` | Passenger's first and last name |
| `Transported` | **Target** — whether the passenger was transported (train only) |

## 🔍 Project Workflow

1. **Basic Data Understanding** — shape, dtypes, duplicates, missing values, cardinality checks
2. **Exploratory Data Analysis (EDA)** — target distribution, age, spending patterns, categorical feature relationships with `Transported`
3. **Feature Engineering**
   - `PassengerId` → `Group_Size`, `Travelling_Solo`
   - `Cabin` → `Cabin_Deck`, `Cabin_Number`, `Cabin_Side`, and binned `Cabin_Region1–6`
   - `Age` → `Age Group` (binned into 6 groups)
   - Spending columns → `Total Expenditure`, `No Spending`, `Expenditure Category`
4. **Data Preprocessing**
   - Missing value imputation (mode for categorical, median for numerical)
   - Dropping high-cardinality raw columns (`PassengerId`, `Cabin`, `Name`)
   - Log-transforming skewed spending features
   - Label Encoding (ordinal features) + One-Hot Encoding (nominal features)
   - Feature scaling with `StandardScaler`
5. **Model Building** — trained and compared 11 classifiers:
   - Logistic Regression, KNN, SVM, Naive Bayes
   - Decision Tree, Random Forest, AdaBoost, Gradient Boosting
   - LightGBM, XGBoost, CatBoost
6. **Hyperparameter Tuning** — `GridSearchCV` on the top performers (LightGBM, CatBoost, XGBoost, Random Forest)
7. **Model Stacking** — a `StackingClassifier` combining the four tuned models for the final prediction
8. **Prediction & Submission** — generates `spaceship_prediction_project.csv` in the Kaggle submission format

## 📊 Results

Gradient-boosted models (LightGBM, XGBoost, CatBoost) and Random Forest outperformed the simpler baselines, with **LightGBM reaching the highest single-model accuracy (~82%)** on the held-out test split. These four models were tuned and combined in a stacking ensemble for the final submission.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn, missingno
- **Modeling:** scikit-learn, XGBoost, LightGBM, CatBoost

## 📁 Project Structure

```
.
├── titanic-spaceship-competition-end-to-end-project.ipynb   # Main notebook
├── train.csv                                                 # Training data (download from Kaggle)
├── test.csv                                                  # Test data (download from Kaggle)
├── spaceship_prediction_project.csv                          # Generated submission file
└── README.md
```

## 📌 Key Takeaways

- Feature engineering (splitting `PassengerId` and `Cabin`, creating spending/age groups) added significant predictive signal beyond the raw features.
- Boosting-based models handled the engineered feature set best.
- A stacking ensemble of the tuned top models gave a well-generalized final result with no strong signs of overfitting.
