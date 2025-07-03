# decision_trees_random_forests

## 📌 Objective
This task focuses on learning tree-based machine learning models — **Decision Tree** and **Random Forest** — for classification using the **Heart Disease Dataset**. The goal is to train, visualize, evaluate, and compare models while understanding concepts like overfitting, feature importance, and cross-validation.

---

## 🧰 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Graphviz / plot_tree for visualization

---

## 📂 Dataset
**Dataset Used:** [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

This dataset contains medical information about patients, including age, cholesterol, blood pressure, etc., to predict the presence of heart disease (`target` column).

---

## 🧪 Steps Performed

### 1. Data Preprocessing
- Loaded and explored the dataset
- Checked for missing values (none found)
- Split data into features and target

### 2. Model Training – Decision Tree
- Trained a default `DecisionTreeClassifier`
- Visualized the tree using `plot_tree`
- Evaluated using accuracy and confusion matrix

### 3. Overfitting Control
- Restricted tree depth using `max_depth=4`
- Compared accuracy and visualized the pruned tree

### 4. Model Training – Random Forest
- Trained a `RandomForestClassifier` with 100 trees
- Compared performance with the decision tree
- Evaluated accuracy, precision, recall, and F1-score

### 5. Feature Importance
- Extracted and plotted feature importances from the random forest model

### 6. Cross-Validation
- Performed 5-fold cross-validation
- Compared the stability and average accuracy of both models

## 📊 Visualizations
- Tree diagrams of full and pruned decision trees
- ![Screenshot 2025-07-03 182559](https://github.com/user-attachments/assets/83534fc3-ae75-4833-a9fe-67e86c29cfb0)
- ![Screenshot 2025-07-03 182626](https://github.com/user-attachments/assets/9ef1b83c-c9e6-430e-adb6-a930f75ee34b)

- Confusion matrices
- ![Screenshot 2025-07-03 183919](https://github.com/user-attachments/assets/88cd768d-d142-4ab7-b8b9-446cae0352ee)
- ![Screenshot 2025-07-03 184009](https://github.com/user-attachments/assets/8a46a0b4-620c-4d5e-806e-c4b326e16b06)

- Bar chart of feature importances
- ![Screenshot 2025-07-03 184026](https://github.com/user-attachments/assets/101272da-4bf2-4edf-b289-a3491963c27f)
