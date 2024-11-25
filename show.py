import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.widgets import Slider

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

A_RELU_FORMAT = r'$\exp( (-{c:.2f}) \cdot D^{{{alpha:.2f}}} + ({b:.2f}) ) + {A_0:.3f}$'
def generate_label_relu(alpha, b, c, A_0):
    return A_RELU_FORMAT.format(alpha=alpha, b=b, c=c, A_0=A_0)
def A_relu(x, alpha, b, c, A_0):
    return np.exp(-c * x ** alpha + b) + A_0

A_SILU_FORMAT = r'${c:.2f} \cdot D^{{-{alpha:.2f}}} + {A_0:.3f}$'
def generate_label_silu(alpha, b, c, A_0):
    return A_SILU_FORMAT.format(alpha=alpha, c=c, A_0=A_0)
def A_silu(x, alpha, b, c, A_0):
    return -c * x ** -alpha + A_0

def load_data(model: str):
    with open('data/{}.txt'.format(model)) as f:
        data = json.loads(f.read())
    x = np.array([value / 1e9 for value in data['token_passed']])
    y = np.array(data['activation_ratio'])
    return x, y

def main(model: str):
    x, y = load_data(model)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(left=0.1, bottom=0.3)

    ax0: Axes = ax[0]
    ax1: Axes = ax[1]

    ax0.scatter(x, y)
    if 'relu' in model:
        ax0.plot(X, A_relu(X, *CURVE_PARAM[model]), label=generate_label_relu(*CURVE_PARAM[model]))
    elif 'silu' in model:
        ax0.plot(X, A_silu(X, *CURVE_PARAM[model]), label=generate_label_silu(*CURVE_PARAM[model]))

    ax0.legend()
    ax0.set_xbound(0, x.max() * 1.2)
    ax0.set_ybound(y.min() - 0.03, y.max() + 0.03)
    ax0.set_xlabel("Token Passed (B)")
    ax0.set_ylabel("Activation Ratio")

    # --- #

    alpha, b, c, A_0 = CURVE_PARAM[model]

    if 'relu' in model:
        ax_A_0 = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        ax_b = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider_A_0 = Slider(ax_A_0, 'A_0', 0, 1, valinit=A_0)
        slider_b = Slider(ax_b, 'b', -5.0, 5.0, valinit=b)

        xx = np.log(x)
        yy = np.log(b - np.log(y - A_0))
        sc = ax1.scatter(xx, yy)
        ax1.plot(X, alpha * X + np.log(c))

        def update(val):
            nA_0 = slider_A_0.val
            nb = slider_b.val

            yy = np.log(nb - np.log(y - nA_0))
            sc.set_offsets(np.c_[xx, yy])
            if any(np.isfinite(yy)):
                max_val = np.max(yy[np.isfinite(yy)])
                min_val = np.min(yy[np.isfinite(yy)])
                dis = max_val - min_val
                max_val += 0.1 * dis
                min_val -= 0.1 * dis
                ax1.set_ybound(min_val, max_val)
            fig.canvas.draw_idle()


        ax1.set_xbound(0, xx.max() * 1.2)
        max_val = np.max(yy[np.isfinite(yy)])
        min_val = np.min(yy[np.isfinite(yy)])
        dis = max_val - min_val
        max_val += 0.1 * dis
        min_val -= 0.1 * dis
        ax1.set_ybound(min_val, max_val)
        ax1.set_xlabel(r"$\ln(D)$")
        ax1.set_ylabel(r"$\ln(b - \ln(A - A_0))$")

        slider_A_0.on_changed(update)
        slider_b.on_changed(update)

    elif 'silu' in model:
        ax_A_0 = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider_A_0 = Slider(ax_A_0, 'A_0', 0, 1, valinit=A_0)

        xx = np.log(x)
        yy = np.log(A_0 - y)
        sc = ax1.scatter(xx, yy)
        ax1.plot(X, -alpha * X + np.log(c))

        def update(val):
            nA_0 = slider_A_0.val
            yy = np.log(nA_0 - y)
            sc.set_offsets(np.c_[xx, yy])
            if any(np.isfinite(yy)):
                max_val = np.max(yy[np.isfinite(yy)])
                min_val = np.min(yy[np.isfinite(yy)])
                dis = max_val - min_val
                max_val += 0.1 * dis
                min_val -= 0.1 * dis
                ax1.set_ybound(min_val, max_val)
            fig.canvas.draw_idle()

        ax1.set_xbound(0, xx.max() * 1.2)
        max_val = np.max(yy[np.isfinite(yy)])
        min_val = np.min(yy[np.isfinite(yy)])
        dis = max_val - min_val
        max_val += 0.1 * dis
        min_val -= 0.1 * dis
        ax1.set_ybound(min_val, max_val)
        ax1.set_xlabel(r"$\ln(D)$")
        ax1.set_ylabel(r"$\ln(A_0 - A)$")
        
        slider_A_0.on_changed(update)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()
    main(args.model)
