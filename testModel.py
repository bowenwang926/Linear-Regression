import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

trainingHours = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(
    -1, 1
)
parsePercentiles = np.array(
    [56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]
).reshape(-1, 1)

time_train, time_test, parse_train, parse_test = train_test_split(
    trainingHours, parsePercentiles, test_size=0.3
)

model = LinearRegression().fit(time_train, parse_train)

print(model.score(time_test, parse_test))

plt.scatter(time_train, parse_train)
plt.plot(
    np.linspace(0, 70, 100).reshape(-1, 1),
    model.predict(np.linspace(0, 70, 100).reshape(-1, 1)),
    "r",
)
plt.show()
