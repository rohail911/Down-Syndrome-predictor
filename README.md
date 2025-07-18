

 Down Syndrome Risk Prediction Tool

This project leverages machine learning, Bayesian statistics, and real-world risk factors to predict the likelihood of Down Syndrome (DS) in newborns. It integrates logistic regression, neural networks (MLP), and Bayesian probability based on maternal/paternal age, carrier status, karyotype abnormalities, and SNP test results.

 Features

* Preprocessing: Cleans and standardizes input data.
* Bayesian Estimation: Computes DS risk using likelihood ratios and prior probabilities.
*  ML Models: Trains:

  * Logistic Regression (with class balancing)
  * Neural Network (MLPClassifier)
*  **SMOTE:** Handles data imbalance via oversampling.
*  **Evaluation:** ROC curves, confusion matrix, classification reports.
*  **Interactive CLI:** Allows real-time DS risk predictions with Bayesian and ML estimates.



 Dataset

The model expects a dataset (in `.xlsx` format) with the following relevant columns:

| Column Name              | Description                                     |
| ------------------------ | ----------------------------------------------- |
| `maternal_age` or `age`  | Age of the mother                               |
| `paternal_age`           | Age of the father                               |
| `mother_carrier_status`  | Whether the mother is a carrier (0/1)           |
| `father_carrier_status`  | Whether the father is a carrier (0/1)           |
| `mother_karyotype`       | Abnormal maternal karyotype (0/1)               |
| `father_karyotype`       | Abnormal paternal karyotype (0/1)               |
| `snp_abnormal`           | SNP abnormality detected (0/1)                  |
| `diagnosis` or `ds_risk` | Target variable: DS diagnosis (0 = No, 1 = Yes) |

🛠 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Required libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn
* openpyxl



 Running the Project

1. Place your dataset at:

```
C:/Users/Jahanzeb/Downloads/ds_data_with_paternal_age.xlsx
```

2. Run the script:

```bash
python your_script_name.py
```

3. You'll be prompted to enter patient data in the terminal:

```bash
Maternal Age: 42
Paternal Age: 47
Mother is Carrier (0/1): 1
Father is Carrier (0/1): 0
Mother Karyotype Abnormal (0/1): 1
Father Karyotype Abnormal (0/1): 0
SNP Abnormality Detected (0/1): 1
```

4. Output:

* Bayesian estimated risk
* Risk from logistic regression
* Risk from neural network

---

 Model Performance

* Logistic Regression & MLPClassifier**

  * Evaluated using confusion matrix, ROC curve, AUC
  * Handles class imbalance using SMOTE



 Bayesian Logic

The Bayesian risk is calculated by adjusting **prior odds** using **likelihood ratios** based on clinical literature and age groups.

| Maternal Age | Risk      |
| ------------ | --------- |
| < 35         | 1 in 1250 |
| 35–39        | 1 in 350  |
| 40–44        | 1 in 100  |
| ≥ 45         | 1 in 30   |

Total posterior risk combines maternal/paternal age, carrier and karyotype status, and SNP results.



 Example Output

```
[Bayesian Estimate] Risk of Down Syndrome: 32.56%
[Logistic Model Estimate] Risk: 41.20%
[Neural Net Estimate] Risk: 38.78%
```





