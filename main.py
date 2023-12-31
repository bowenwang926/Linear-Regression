import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


trainingHours = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(
    -1, 1
)
parsePercentiles = np.array(
    [56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]
).reshape(-1, 1)

model = LinearRegression().fit(trainingHours, parsePercentiles)

print(model.predict(np.array([56]).reshape(-1, 1)))

plt.scatter(trainingHours, parsePercentiles)
plt.plot(
    np.linspace(0, 70, 100).reshape(-1, 1),
    model.predict(np.linspace(0, 70, 100).reshape(-1, 1)),
    "r",
)
plt.ylim(0, 100)
plt.show()
