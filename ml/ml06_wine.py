import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# 데이터 읽어 들이기
wine = pd.read_csv('./data/winequality-white.csv',
                   sep=";", encoding='utf-8')

# 데이터를 레이블과 데이터로 분리하기
y = wine['quality']
x = wine.drop('quality', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 모델 구성
model = RandomForestClassifier()

# 훈련
model.fit(x_train, y_train)

'''
keras에는 model fit, evaluate, predict가 있지만
sklearn에는 evaluate가 없고 score가 있다
'''

# 평가 예측 
aaa = model.score(x_test, y_test) # loss를 뺀 accuracy만 제공
print('aaa : ', aaa)

y_pred = model.predict(x_test)
print('정답률 : ', accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))