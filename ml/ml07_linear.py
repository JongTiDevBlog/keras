from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

boston = load_boston()

x = boston.data
y = boston.target

from sklearn.linear_model import LinearRegression, Ridge, Lasso

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test) # loss를 뺀 accuracy만 제공
print('aaa : ', aaa)

model = Ridge()
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test) # loss를 뺀 accuracy만 제공
print('aaa : ', aaa)

model = Lasso()
model.fit(x_train, y_train)
aaa = model.score(x_test, y_test) # loss를 뺀 accuracy만 제공
print('aaa : ', aaa)
