import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from scipy.optimize import curve_fit


# our own fitted curve in the paper
X = np.linspace(0, 400, 1000)
CURVE_PARAM = {
    '0.1b_relu' : (1.01e-1, -1.51e-2, 3.20e+0, 6.14e-2),
    '0.2b_relu' : (4.49e-1, -3.05e+0, 2.86e-1, 6.74e-2),
    '0.4b_relu' : (6.83e-1, -3.46e+0, 7.90e-2, 6.90e-2),
    '0.8b_relu' : (1.01e+0, -3.49e+0, 7.97e-3, 7.21e-2),
    '1.2b_relu' : (1.33e+0, -3.89e+0, 9.03e-4, 7.82e-2),
    '2.4b_relu' : (2.11e+0, -3.82e+0, 9.04e-6, 7.12e-2),
    
    '0.1b_silu' : (4.79e-1, None, 1.02e-1, 4.09e-1),
    '0.2b_silu' : (8.44e-1, None, 2.08e-1, 3.90e-1),
    '0.4b_silu' : (1.03e+0, None, 4.20e-1, 3.85e-1),
    '0.8b_silu' : (9.95e-1, None, 5.62e-1, 3.83e-1),
    '1.2b_silu' : (9.67e-1, None, 5.38e-1, 3.82e-1),
}

def A_relu(x, alpha, b, c, A_0):
    return np.exp(-c * x ** alpha + b) + A_0
def A_silu(x, alpha, b, c, A_0):
    return -c * x ** -alpha + A_0

def test_func(x, a, b, A_0):
    return np.log(a * x + b) + A_0

def load_data(model: str):
    with open('data/{}.txt'.format(model)) as f:
        data = json.loads(f.read())
    x = np.array([value / 1e9 for value in data['token_passed']])
    y = np.array(data['activation_ratio'])
    return x, y

def main(model: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax: Axes

    x, y = load_data(model)
    if 'relu' in model:
        y_pred = A_relu(x, *CURVE_PARAM[model])
        ax.plot(X, A_relu(X, *CURVE_PARAM[model]))
    elif 'silu' in model:
        y_pred = A_silu(x, *CURVE_PARAM[model])
        ax.plot(X, A_silu(X, *CURVE_PARAM[model]))

    mae_power, mse_power = np.average(np.abs(y - y_pred)), np.average(np.square(y - y_pred))

    popt, _ = curve_fit(test_func, x, y, maxfev=40000, p0=(np.random.rand(), np.random.rand() * 10, 0))
    y_pred = test_func(x, *popt)
    mae_log, mse_log = np.average(np.abs(y - y_pred)), np.average(np.square(y - y_pred))
    print("fitted log coefficients:", *popt)

    print("MAE of power-law:", mae_power)
    print("MAE of logarithmic function:", mae_log)
    print("MSE of power-law:", mse_power)
    print("MSE of logarithmic function:", mse_log)

    ax.scatter(x, y)
    ax.plot(X, test_func(X, *popt))
    ax.set_xbound(0, x.max() * 1.2)
    ax.set_ybound(y.min() - 0.03, y.max() + 0.03)
    ax.set_xlabel("Token Passed (B)")
    ax.set_ylabel("Activation Ratio")

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    main(args.model)