**Bank Marketing Classification – Machine Learning Assignment 2**

1. **Problem Statement**: The objective of this project is to build and compare multiple machine learning classification models to predict whether a bank customer will subscribe to a  
   term deposit based on demographic, financial, and marketing campaign attributes.  
     
   This is a binary classification problem where the target variable indicates subscription (\`yes\` or \`no\`).  
     
2. **Dataset Description:** The dataset used is the **Bank Marketing Dataset** obtained from the

UCI Machine Learning Repository.

\- Instances: 45,211  
\- Features: 16 input features  
\- Target variable: \`y\` (yes/no)  
\- Dataset type: Binary classification

The dataset contains both numerical and categorical attributes related to direct marketing campaigns conducted by a Portuguese banking institution. Missing values in categorical attributes represent unknown information and were treated as a separate category during preprocessing.

3. **Models Used and Evaluation Metrics:**   
   The following six machine learning classification models were implemented on the same dataset:  
     
   \- Logistic Regression  
   \- Decision Tree Classifier  
   \- K-Nearest Neighbors (KNN)  
   \- Naive Bayes (Gaussian)  
   \- Random Forest (Ensemble)  
   \- XGBoost (Ensemble)

   Each model’s implementation is provided as a Python script in the models/ directory, while trained models are stored as serialized .pkl files for deployment.

   

   

   

   

   

   

   

   

   **Model Performance Comparison**

   

   

| ML Model Name  | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Logistic Regression** | 0.8914 | 0.8726 | 0.5945 | 0.2259 | 0.3274 | 0.3205 |
| **Decision Tree** | 0.8970 | 0.8374 | 0.5893 | 0.3960 | 0.4737 | 0.4293 |
| **kNN** | 0.8923 | 0.8089 | 0.5717 | 0.3166 | 0.4075 | 0.3724 |
| **Naive Bayes** | 0.8380 | 0.8127 | 0.3554 | 0.4726 | 0.4057 | 0.3183 |
| **Random Forest (Ensemble)**  | 0.9063 |  0.9246 | 0.6572 | 0.4168 | 0.5101 | 0.4758 |
| **XGBoost (Ensemble)** | 0.9058 | 0.9267 | 0.6281 | 0.4773 | 0.5424 | 0.4968 |

   

**Observations on Model Performance**

| ML Model Name  | Observations |
| :---- | ----- |
| **Logistic Regression** | Achieved high accuracy but low recall, indicating difficulty in capturing minority positive class instances. |
| **Decision Tree** | Performed better than logistic regression in recall and F1 score by capturing non-linear relationships. |
| **kNN** | Showed moderate performance but was sensitive to feature scaling and neighborhood size**.** |
| **Naive Bayes** | Demonstrated higher recall but lower precision due to strong independence assumptions. |
| **Random Forest (Ensemble)**  | Improved overall performance by reducing overfitting and variance through ensemble learning. |
| **XGBoost (Ensemble)** | Achieved the best overall balance across metrics, with the highest AUC and MCC, indicating superior class discrimination. |

