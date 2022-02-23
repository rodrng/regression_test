import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

body_data = pd.read_csv('data/weight-height.csv')

regression_model = sm.OLS.from_formula("Weight~Height", body_data).fit()

print(regression_model.summary())

height = body_data['Height'].tolist()
weight = body_data['Weight'].tolist()

body = pd.DataFrame({'몸무게':weight, '키':height})

plt.scatter(body['키'],body['몸무게'])

plt.xlabel('HEIGHT')
plt.ylabel('WEIGHT')

plt.show()