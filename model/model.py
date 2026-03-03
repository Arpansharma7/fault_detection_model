import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv("dataset/TRAIN.csv")
test = pd.read_csv("dataset/TEST.csv")

# Separating feature and target
X = train.drop(columns=["Class"])
Y = train["Class"]

test_ids = test["ID"]
X_test = test.drop(columns=["ID"])

#handling missing values
X = X.fillna(X.median())
X_test = X_test.fillna(X.median())

#cross validation
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
auc_scores = []

for train_idx, val_idx in kf.split(X, Y):
    X_tr, X_val = X.iloc[train_idx],X.iloc[val_idx]
    Y_tr,Y_val=Y.iloc[train_idx], Y.iloc[val_idx]

    model = LGBMClassifier(
        n_estimators=4000, learning_rate=0.02,
        num_leaves=63, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
        verbosity=-1
    )
    model.fit(
        X_tr, Y_tr,
        eval_set=[(X_val, Y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100)]
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(Y_val, preds)
    auc_scores.append(auc)
print("AUC:", auc_scores)
print("Average auc score:",np.mean(auc_scores))

#training on full dataset 
final_model = LGBMClassifier(
    n_estimators=model.best_iteration_,
    learning_rate=0.02,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
final_model.fit(X, Y)
final_preds = final_model.predict(X_test)
out = pd.DataFrame({
    "ID": test_ids,
    "CLASS": final_preds
})
out.to_csv("FINAL.csv", index=False)
