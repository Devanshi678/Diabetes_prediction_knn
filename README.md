# Diabetes_prediction_knn
# 🩺 Diabetes Prediction using K-Nearest Neighbors (KNN)

This project is a machine learning classification model built to predict the likelihood of diabetes in patients using the **Pima Indians Diabetes Dataset**. The project was originally created during my undergraduate internship and rebuilt using Python, scikit-learn, and data mining techniques. The K-Nearest Neighbors (KNN) algorithm is used for the prediction task.

---

## 📊 Dataset Description

The dataset contains medical diagnostic measurements for 768 patients. It includes 8 features and 1 target (Outcome):

| Feature                        | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| Pregnancies                   | Number of times the patient was pregnant                                   |
| Glucose                       | Plasma glucose concentration (mg/dL) after 2 hours in OGTT                 |
| BloodPressure                 | Diastolic blood pressure (mm Hg)                                           |
| SkinThickness                 | Triceps skin fold thickness (mm)                                           |
| Insulin                       | 2-Hour serum insulin (mu U/ml)                                             |
| BMI                           | Body Mass Index: weight (kg) / height (m²)                                 |
| DiabetesPedigreeFunction      | Genetic relationship measure with other diabetic patients                  |
| Age                           | Age of the patient (years)                                                 |
| **Outcome**                   | 1 = Diabetic, 0 = Not Diabetic (Target variable)                           |

---

## 🧠 Machine Learning Approach

- Algorithm: **K-Nearest Neighbors (KNN)**
- Preprocessing:
  - Data Cleaning & Description (`.describe()`)
  - Train-Test Split (80% train, 20% test)
  - Feature Standardization using `StandardScaler`
- Model Training:
  - Default `KNeighborsClassifier(n_neighbors=5, weights='uniform')`
- Model Evaluation:
  - Accuracy Score
  - Confusion Matrix
  - Precision, Recall, F1 Score
  - Classification Report

---

## 📈 Results

- ✅ **Model Accuracy**: ~74%
- 📋 **Performance Metrics**:
  - Balanced classification performance
  - Precision, recall, and F1-score analyzed for both diabetic and non-diabetic classes
- 🔍 **Most Influential Features** (based on correlation analysis):
  - BMI
  - Age
  - Glucose

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas**, **numpy**
- **scikit-learn**
- **matplotlib**, **seaborn**

---

## 🚀 How to Run

1. Clone the repository or download the project files
2. Install required packages:
   ```bash
   pip install -r requirements.txt
