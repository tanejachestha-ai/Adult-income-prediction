Census Income Classification (Machine Learning Project)

Project Overview
This project aims to predict whether an individual earns **more than \$50,000 per year** based on various demographic and employment features from the **U.S. Census Income dataset**.  
Multiple machine learning classification models were trained and compared to determine the most accurate model for income prediction.

---

Goal
To build a classification model that predicts income category (≤50K or >50K) and identify key factors influencing income levels.

---

Machine Learning Models Used
- **Logistic Regression**
- **K-Nearest Neighbors (KNN) Classifier**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Naive Bayes Classifier**

Best Model:** Logistic Regression  
Best Accuracy:** Achieved high accuracy with balanced performance across both income classes.

---

Technologies & Libraries
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **IDE:** Jupyter Notebook  

---

Steps Performed
1. **Data Loading & Exploration:**  
   Imported and explored the Census Income dataset to understand feature types and distributions.
2. **Data Cleaning & Preprocessing:**  
   - Removed missing and irrelevant values  
   - Encoded categorical variables using Label Encoding / OneHotEncoding  
   - Standardized numerical features
3. **Train-Test Split:**  
   Split dataset into training and testing sets using `train_test_split()`.
4. **Model Training:**  
   Trained multiple classification algorithms on the dataset.
5. **Model Evaluation:**  
   Compared accuracy, precision, recall, and F1-score to determine the best-performing model.

---

Model Comparison

| Algorithm | Accuracy | Remarks |
|------------|-----------|---------|
| Logistic Regression | **Highest (Best Model)** | Balanced & interpretable |
| KNN Classifier | Moderate | Sensitive to scaling |
| Decision Tree | High | Slightly overfitted |
| Random Forest | High | Strong ensemble performance |
| Naive Bayes | Good | Works well with categorical data |

---

Key Learnings
- How to preprocess mixed data types (categorical + numerical).  
- Comparison of different classification algorithms.  
- Importance of feature scaling and encoding.  
- Model evaluation using multiple metrics (Accuracy, Precision, Recall, F1-score).  
- Understanding how demographic and occupational features influence income.
