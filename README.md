**This repository contains my solution to the problem for fault detection given for ML alrIEEEna - IEEE GEHU:**

*PROBLEM OVERVIEW:*

  47 features
  
  target label : Class (0,1)
  
*OBJECTIVE:*

  to train a model using TRAIN.csv and generate predictions for TEST.csv in the given format

*APPROACH:*

  1.data preparation
    seperated features and target
    removed id column from test dataset
    handled missing data(if there was)
  
  2. Model selection:
     i use LightGBM (gradient boosting) because:
       the dataset is numerical and structured
       it captures the non linear realtions effectively
  3. Validation:
     used 5 fold cross validation
     evalaution metric : ROC-AUC
     also applied early stopping
  4.output:
    model trained on full dataset
    saved output as FINAL.csv
    with cross-validation ~ 0.9993


*PROJECT STRUCTURE:*

```
fault_detection/
│
├── dataset/
│   ├── TRAIN.csv
│   └── TEST.csv
│
├── output/
│   └── FINAL.csv
│
├── model.py
├── requirements.txt
└── README.md
```
*HOW TO RUN:*

  Install dependencies:
  
    pip install -r requirements.txt

  Run:

    python model.py

  The predictions will be saved as:

    FINAL.csv

*Libraries used:*
  ```
  pandas
  lightgbm
  scikit-learn
  ```
