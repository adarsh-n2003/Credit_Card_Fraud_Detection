### **Credit Card Fraud Detection Project**

This repository contains code for a fraud detection project aimed at identifying potentially fraudulent transactions in credit card data using machine learning algorithms.

### **Dataset**

The dataset used in this project is obtained from Kaggle and is provided by the Machine Learning Group at ULB (Université Libre de Bruxelles). It contains transactions made by credit cards in September 2013 by European cardholders. The dataset consists of 284,807 transactions, of which 492 are fraudulent. The dataset is highly imbalanced, with fraudulent transactions accounting for only 0.172% of all transactions.

You can access the dataset on Kaggle via the following link: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### **Project Overview**

The project involves the following steps:

1. **Exploratory Data Analysis (EDA):** Analyzing the dataset to understand its structure, check for missing values, examine the distribution of features, and visualize the distribution of the target variable (fraudulent vs. non-fraudulent transactions).
2. **Model Building:** Constructing a machine learning model using the Random Forest Classifier algorithm to predict whether a transaction is fraudulent or not.
3. **Model Evaluation:** Evaluating the performance of the model using metrics such as accuracy score and classification report.

### **Files**

- **`fraud_detection.py`**: Python script containing the code for EDA, model building, and evaluation.
- **`credit_card_dataset.csv`**: CSV file containing the credit card transaction data.

### **Usage**

To use the code in this repository:

1. Clone the repository to your local machine.
2. Download the **`credit_card_dataset.csv`** file from the provided Kaggle link and place it in the repository directory.
3. Run the **`fraud_detection.py`** script to execute the project code.

### **Requirements**

The project code requires the following Python libraries:

- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these dependencies using pip:

```
Copy code
pip install pandas matplotlib seaborn scikit-learn

```

### **Contributors**

- [Adarsh Nashine](https://www.linkedin.com/in/adarsh-nashine/)

### **Acknowledgments**

- The dataset used in this project is sourced from Kaggle and provided by the Machine Learning Group at ULB (Université Libre de Bruxelles).
- Special thanks to the contributors and maintainers of the dataset for making it publicly available for research purposes.

### **License**

This project is licensed under the MIT License. Feel free to use the code for educational and commercial purposes.
