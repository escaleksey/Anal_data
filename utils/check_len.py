import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

class CheckLength():
    def ci_mean(self, df:pd.DataFrame, alpha=0.05):
        print("Доверительный интервал")
        res = []
        for col in df.columns:
            data = df[col]
            n = len(data)
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_val * (std / np.sqrt(n))
            res.append((col, round(float(mean - margin),3), round(float(mean + margin), 3)))
        return res

    def plot_mean_stability(self, df):
        for col in df.columns:
            data = df[col]
            means = [data[:i].mean() for i in range(10, len(data), 3)]
            plt.plot(range(10, len(data), 3), means)
            plt.axhline(data.mean(), color='red', linestyle='--', label='Итоговое среднее')
            plt.title("Сходимость среднего при росте выборки")
            plt.xlabel("Размер выборки")
            plt.ylabel("Среднее")
            plt.legend()
            plt.show()