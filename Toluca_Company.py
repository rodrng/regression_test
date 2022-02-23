import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

toluca_data = pd.read_csv('data/toluca_company_dataset.csv')
print(toluca_data)

# 두 변수의 산점도를 시각화하여 회귀분석 가능 여부 확인


toluca_fit = ols('Work_hours~Lot_size', data=toluca_data).fit() # 회귀분석모델의 적합성 확인
# print(toluca_fit.summary())

# print(toluca_fit.params.Intercept) # 절편
# print(toluca_fit.params.Lot_size) # 기울기

toluca_values = toluca_fit.fittedvalues # 회귀모델에서 측정한 작업시간 추정값
print('---------------------예측모델------------------------')
print(toluca_values)

plt.scatter(toluca_data['Lot_size'],toluca_data['Work_hours'])
plt.plot(toluca_data['Lot_size'], toluca_values, color='red')
plt.xlabel('Lot Size')
plt.ylabel('Work hours')
plt.show()

toluca_predict = toluca_fit.predict(exog=dict(Lot_size=[150])) # 새로운 제품 크기에 대한 작업시간 예측값

# print(toluca_predict)
print('')
print(toluca_fit.resid) # 잔차(실제 데이터 값 회귀모델간의 예측값과의 오차)

# 잔차도 시각화
# plt.scatter(toluca_data['Lot_size'], toluca_fit.resid)
# plt.show()