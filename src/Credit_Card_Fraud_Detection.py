import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score, ConfusionMatrixDisplay
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# Find the creditcard.csv
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

# Remove duplicate transactions to reduce bias in evaluation
df = df.drop_duplicates()
print("Shape after drop_duplicates:", df.shape)
print()

# Amount distribution + outliers
plt.figure(figsize=(7,4))
sns.histplot(df["Amount"], bins=100, log_scale=True)
plt.title("Transaction Amount Distribution (log scale)")
plt.show()


print("\nClass distribution:")
print(df['Class'].value_counts())
print()


sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: No Fraud, 1: Fraud)')
plt.show()

print()
print(df.groupby('Class')['Amount'].describe())


X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
    )


cv = StratifiedKFold(
    n_splits=5,      
    shuffle=True,    
    random_state=42  
)

scoring={           
        'roc_auc': 'roc_auc',               
        'pr_auc': 'average_precision'         
    }

pipeline_smote = Pipeline(steps=[
    
    ('smote', SMOTE(random_state=42, sampling_strategy=0.20)),
    ('model', RandomForestClassifier(
        n_estimators=200,   
        random_state=42,   
        n_jobs=-1,
    ))
])

scores_smote = cross_validate(
    pipeline_smote,          
    X_train,           
    y_train,          
    cv=cv,
    scoring=scoring,
    return_train_score=True  
)

    # Without SMOTE
pipeline_no_smote = Pipeline(steps=[
    ('model', RandomForestClassifier(
        n_estimators=200,   
        random_state=42,    
        n_jobs=-1,           
        class_weight="balanced"   
    ))
])

scores_no_smote = cross_validate(
    pipeline_no_smote,           
    X_train,           
    y_train,            
    cv=cv,
    scoring=scoring,
    return_train_score=True 
)

def summarize(scores_dict):
    return {
        "CV ROC-AUC mean": float(np.mean(scores_dict["test_roc_auc"])),
        "CV ROC-AUC std":  float(np.std(scores_dict["test_roc_auc"])),
        "CV PR-AUC mean":  float(np.mean(scores_dict["test_pr_auc"])),
        "CV PR-AUC std":   float(np.std(scores_dict["test_pr_auc"])),
    }

results = pd.DataFrame([
    {"Approach": "RandomForest + SMOTE", **summarize(scores_smote)},
    {"Approach": 'RandomForest + class_weight="balanced" (no SMOTE)', **summarize(scores_no_smote)},
]).sort_values("CV PR-AUC mean", ascending=False)

results.reset_index(drop=True, inplace=True)


results_rounded = results.copy()
for col in results_rounded.columns[1:]:
    results_rounded[col] = results_rounded[col].round(6)

print(results_rounded)   
print()

print("\nModel selection (based on cross-validation):")
print("I chose the SMOTE pipeline because it achieved a higher mean PR-AUC (Average Precision) than using class_weight='balanced'.")
print("PR-AUC is the primary metric here due to the strong class imbalance in fraud detection.\n")


final_pipeline = pipeline_smote

final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)[:, 1]

print("---- Test Results (Unseen Test Set) ----")
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Test PR-AUC :", average_precision_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, normalize="true", values_format=".4f"
)
plt.title("Confusion Matrix (Normalized)")
plt.show()


RUN_LGBM = False  # set True only if you want to run this section

if RUN_LGBM:
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_validate

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    
    lgbm = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    
    scores_lgbm = cross_validate(
        lgbm, X_train, y_train, cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        return_train_score=False
    )
    
    print("LGBM CV ROC-AUC mean:", scores_lgbm["test_roc_auc"].mean())
    print("LGBM CV PR-AUC mean :", scores_lgbm["test_pr_auc"].mean())
