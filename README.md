# 🩺 Diabetes Prediction Using K-Nearest Neighbors (KNN)

This project applies supervised machine learning to predict the likelihood of diabetes in patients based on health-related metrics. Built using the K-Nearest Neighbors (KNN) algorithm, it explores model tuning and evaluates performance using real-world health data.

---

## 📁 Dataset Overview

The dataset contains information about **768 patients**, including 8 medical features and 1 binary target:

| Feature                        | Description |
|-------------------------------|-------------|
| Pregnancies                   | Number of pregnancies |
| Glucose                       | Plasma glucose concentration (2 hours after a glucose tolerance test) |
| BloodPressure                 | Diastolic blood pressure (mm Hg) |
| SkinThickness                 | Triceps skinfold thickness (mm) |
| Insulin                       | Serum insulin (mu U/ml) |
| BMI                           | Body mass index (weight in kg/(height in m)^2) |
| DiabetesPedigreeFunction      | Diabetes pedigree function (family history) |
| Age                           | Age (years) |
| Outcome                       | 1 if diabetic, 0 if not |

---

## 🧠 Machine Learning Approach

### ✅ Algorithm:
- **K-Nearest Neighbors (KNN)** classification

### ✅ Workflow:
1. Data Cleaning & Normalization
2. Train-Test Split (80%-20%)
3. Model Training with different `k` values (1–20)
4. Plotting **Train vs Test Accuracy** to detect overfitting
5. Selecting optimal `k` (in this case, **k = 18**)
6. Final Model Evaluation (accuracy, confusion matrix, classification report)
7. Optional: Permutation-based Feature Importance

---

## 📊 Model Performance (k = 18)

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | ~76%      |
| Precision    | 0.77      |
| Recall       | 0.67      |
| F1-Score     | 0.72      |

The classifier performs well, especially in identifying non-diabetic patients. Future work may include balancing the dataset or using ensemble models to improve recall.

---

## 🖼️ Visualizations

- 📈 Accuracy vs. `k` plot (to tune hyperparameters)
- 🔥 Confusion Matrix (for classification insight)
- 📌 Permutation-based Feature Importance

---

## 📦 Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## 🚀 Key Takeaways

- **Model tuning matters** — Default `k=5` isn't always optimal
- **Train vs Test Accuracy** helps identify overfitting or underfitting
- **Feature scaling** is essential in KNN
- **KNN is simple but interpretable**, and good for baseline comparisons

---

## 🔮 Future Improvements

- Use **cross-validation** instead of single train/test split
- Try **other algorithms**: Logistic Regression, Random Forest
- Apply **SMOTE or class balancing techniques** to improve recall
- Build a **streamlit dashboard** for real-time prediction

---

## 👨‍🎓 About Me

This project was created as part of my college coursework and later revisited during my Master's program. It demonstrates my ability to apply data mining and machine learning concepts in real-world health prediction problems.

I'm currently seeking **co-op or internship opportunities** in machine learning, data science, or AI!

Feel free to reach out or check out my [LinkedIn](#) 😊

---

## 📁 How to Run

1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebook: `diabetes_knn.ipynb`

---

## 📝 License

This project is under the MIT License. You may use or modify with attribution.
