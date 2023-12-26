from lightgbm import train, Dataset, LGBMClassifier
from ordinalgbt.lgb import LGBMOrdinal
from ordinalgbt.data import simplest_case, make_ordinal_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X,y = make_ordinal_classification(n_classes=4, n_samples=1000, n_features = 100, n_informative =10,noise=1)
X_train,X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)
model = LGBMOrdinal(n_estimators=1001)
model.fit(X=X_train,y = y_train)

normal_model = LGBMClassifier(n_estimators = 1)
normal_model.fit(X,y)
accuracy_score(y,normal_model.predict(X))
