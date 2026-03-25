import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


data = pd.read_csv("Dataset/finetune/eval_512_test_pred_2.csv")
#print(data.columns)
if "example_id" in data:
    agg_probas = data.groupby("example_id")[["prob_0", "prob_1"]].mean()
    pred = (agg_probas["prob_1"] > agg_probas["prob_0"]).astype(int).map({0:"N", 1:"P"})
pred = data["pred_label"].map({0:"N", 1:"P"})

true_labels = pd.read_csv("Dataset/submissions/submission-movie-1.csv", header=None)
print(pred.head(20))
print(accuracy_score(true_labels, pred))
print(precision_score(true_labels, pred, pos_label="P"))
print(recall_score(true_labels, pred, pos_label="P"))
print(f1_score(true_labels, pred, pos_label="P"))

#pred.to_csv("Dataset/submissions/submission-movies-2.csv", index=False, header=None)
#print(pred.head(10), true_labels.head(10))

#true_labels =  data.groupby("example_id")["true_label"].first()

# print(accuracy_score(pred, true_labels))
# print(f1_score(pred, true_labels))
# print(precision_score(pred, true_labels))
# print(recall_score(pred, true_labels))
