from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

data = fetch_covtype(download_if_missing=True)
X = data.data
y = data.target - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=7,
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=300,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
joblib.dump(model,"test.pkl")
