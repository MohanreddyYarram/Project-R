import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

np.random.seed(5)
x = np.random.rand(100,1)*10
y = 5*x+1+np.random.rand(100,1)*2

model = LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

print(f"Slope (Coefficient):{model.coef_[0]}")
print(f"Intercept:{model.intercept_}")
plt.scatter(x,y,color='blue',label='Data points')
plt.plot(x,y_pred,color='green',label='Linear regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Linear Regression Example")
plt.legend()
plt.show()