from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from joblib import dump

df = pd.read_csv("/media/jai/Deck/projects/ai-churn/CIS579_Churn/data/bs_eda_wo_index.csv")

features = ['Senior Citizen', 'Partner', 'Dependents', 'Tenure', 'Phone Service','Multiple Lines', 'Online Security', 'Online Backup','Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies','Paperless Billing', 'Monthly Charges', 'Total Charges','Internet Service_DSL', 'Internet Service_Fiber optic','Internet Service_No', 'Contract_Month-to-month', 'Contract_One year','Contract_Two year', 'Payment Method_Bank transfer (automatic)','Payment Method_Credit card (automatic)','Payment Method_Electronic check', 'Payment Method_Mailed check']
target = ['Churn']
data=[features, target]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2)

# create model instance
bst = XGBClassifier(n_estimators=25, learning_rate=0.0001, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
# y_pred = bst.predict(X_test)

#prediction and Classification Report
from sklearn.metrics import classification_report

pred1 = bst.predict(X_test)
pred2 = bst.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.4f}")
print('Model 1 XGboost Report\n', (classification_report(y_test, pred1)))
print('\nModel 2 XGboost Report\n', (classification_report(y_test, pred2)))

dump(bst, 'xgc1.joblib')