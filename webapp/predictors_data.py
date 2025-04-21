import torch.nn as nn
import torch
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.fc3(x)
        return x

# so that torch thinks this is a safe pickle (just found this on google)
torch.serialization.add_safe_globals([ChurnModel])

ann_model = ChurnModel(input_dim=25)
ann_model.load_state_dict(torch.load(
    "../bharath/ann_weights_only.pth",
    map_location=torch.device('cpu') #so that it works without the gpu too
))
ann_model.eval()

rfr = load("../Stayner/random_forest_churn_model_smote.pkl")
xgc = load("../jaivanth/xgc_gc.joblib")




def predict_all():
    input_df = pd.read_csv("../data/bs_eda_wo_index.csv")
    true = input_df["Churn"].values
    input_df = input_df.drop("Churn", axis=1).astype(float)

    # Neural Net Prediction
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32)
    with torch.no_grad():
        output = ann_model(input_tensor)
    ann_probs = torch.sigmoid(output).numpy().flatten()  # shape: (n_samples,)

    # XGBoost Prediction (probabilities)
    xgc_probs = xgc.predict_proba(input_df)[:, 1]  # Only positive class probability

    # Random Forest Regression (already probability-like)
    rfr_probs = rfr.predict(input_df)  # shape: (n_samples,)

    # Average ensemble
    avg_probs = (ann_probs + xgc_probs + rfr_probs) / 3
    pred_labels = (avg_probs >= 0.5).astype(int)

    print(classification_report(true.astype(int), pred_labels))

    # Confusion Matrix
    cm = confusion_matrix(true.astype(int), pred_labels)
    classes = ["No Churn", "Churn"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()



predict_all()