from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x = np.array([121, 125, 131, 141, 152, 161]).reshape(-1,1) # x denotes the house area as a feature.
y = np.array([300, 350, 425, 405, 496, 517]) # y denotes the house price.
plt.scatter(x,y)
plt.xlabel("area") # X axis indicates the area.
plt.ylabel("price") # Y axis indicates the price.
plt.show()

lr = LinearRegression() # Encapsulate the linear regression model into an object.
lr.fit(x,y) # Train the model on the dataset.

w = lr.coef_
b = lr.intercept_
print('Slope: ',w)
print('Intercept: ',b)

plt.scatter(x,y)
plt.xlabel("area")
plt.ylabel("price")
plt.plot([x[0],x[-1]],[x[0]*w+b,x[-1]*w+b])
plt.show()

testX = np.array([[130]])
print(lr.predict(testX))