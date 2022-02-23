import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression # 선형회귀 모델
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

mpg_df = pd.read_csv('data/auto-mpg.csv')
# print(mpg_df)

mpg_df1 = mpg_df.drop(['horsepower', 'origin', 'car_name'], axis=1) # 필요없는 변수 제거 (제외시킬 때 drop + 제외시킬 명 + 축)
# print(mpg_df1.info())

y = mpg_df1['mpg'] # 종속변수(연비)
x = mpg_df1.drop(['mpg'], axis=1) # mpg(종속변수)를 제외한 독립변수들

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0) # 훈련용 데이터와 평가용 데이터 분할

lr = LinearRegression() # 회귀모델 생성
lr.fit(X_train, Y_train) # 모델 훈련

Y_predict = lr.predict((X_test)) # 평가데이터에 대한 예측 수행

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
r2_s = r2_score(Y_test, Y_predict)

print('mse:',mse)
print('rmse:',rmse)
print('결정계수:',r2_s)

mpg_intercept = lr.intercept_
mpg_coef = lr.coef_

print('절편:', mpg_intercept)
print('회기계수:', mpg_coef)

coef = pd.Series(data=mpg_coef, index=x.columns)
print(coef)

import matplotlib.pyplot as plt
import seaborn as sns

# fig, axs = plt.subplots(figsize=(15,15), ncols=3, nrows=2)
# x_fea = ['cylinders', 'displacement', 'weight', 'acceleration', 'model_year']
#
# sns.regplot(x_fea[0], y='mpg', data=mpg_df1, ax=axs[0][0])
# sns.regplot(x_fea[1], y='mpg', data=mpg_df1, ax=axs[0][1])
# sns.regplot(x_fea[2], y='mpg', data=mpg_df1, ax=axs[0][2])
# sns.regplot(x_fea[3], y='mpg', data=mpg_df1, ax=axs[1][0])
# sns.regplot(x_fea[4], y='mpg', data=mpg_df1, ax=axs[1][1])

# plt.show()
print('연비를 예측하고 싶은 자동차의 정보를 입력하세요.')

cylinders_ = input('실린더수를 입력하세요:')
cylinders_ = int(cylinders_)
displacement_ = input('배기량을 입력하세요:')
displacement_ = int(displacement_)
weight_ = input('차의 무게를 입력하세요:')
weight_ = int(weight_)
acceleration_ = input('가속력을 입력하세요:')
acceleration_ = int(acceleration_)
model_year_ = input('차의 연식을 입력하세요:')
model_year_ = int(model_year_)

mpg_predict = lr.predict([[cylinders_, displacement_, weight_, acceleration_, model_year_]])
print('입력하신 자동차의 연비(mpg)는 %f 입니다.' %mpg_predict)
