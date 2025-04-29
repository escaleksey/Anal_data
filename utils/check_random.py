from scipy.stats import kendalltau, norm, shapiro, anderson

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.runs import runstest_1samp
from statsmodels.graphics.tsaplots import plot_acf

from scipy.stats import chi2, norm, expon


class CheckRandom():

    def check_pirs(self,
                   df: pd.DataFrame,
                   dist=norm,
                   bins='sturges',
                   param_estimator=None):
        """
        Применяет χ²-тест согласия Пирсона ко всем числовым столбцам DataFrame.

        Параметры:
        - df: pd.DataFrame — таблица с числовыми данными.
        - dist — распределение из scipy.stats (по умолчанию: нормальное).
        - bins: int или 'sturges' — число бинов или признак использования формулы Стерджеса.
        - param_estimator — функция: pd.Series → tuple(params). Если None, используется (mean, std).

        Возвращает DataFrame с результатами по столбцам.
        """
        if param_estimator is None:
            param_estimator = lambda s: (s.mean(), s.std(ddof=0))

        results = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            n = len(series)
            if n == 0:
                continue

            # Оценка параметров
            params = param_estimator(series)
            x = series.values

            # Выбор бинов
            if bins == 'sturges':
                k = 1 + 3.32 * np.log10(n)
                delta = (x.max() - x.min()) / k
                # Построение границ от min до max с шагом delta
                edges = np.arange(x.min(), x.max() + delta, delta)
            else:
                k = int(bins)
                probs = np.linspace(0, 1, k + 1)
                edges = dist.ppf(probs, *params)
                edges[0] = min(edges[0], x.min() - 1e-6)
                edges[-1] = max(edges[-1], x.max() + 1e-6)

            # Наблюдаемые частоты
            obs_counts, _ = np.histogram(x, bins=edges)
            # Ожидаемые частоты
            cdf_vals = dist.cdf(edges, *params)
            exp_counts = n * np.diff(cdf_vals)

            # Статистика χ²
            chi2_stat = ((obs_counts - exp_counts) ** 2 / exp_counts).sum()
            dfree = len(edges) - 1 - len(params)
            p_value = 1 - chi2.cdf(chi2_stat, dfree)

            note = ''
            if np.any(exp_counts < 5):
                note = 'E_i < 5'

            results.append({
                'column': col,
                'chi2_stat': round(chi2_stat, 3),
                'p_value': round(p_value, 4),
                'df': dfree,
                'note': note
            })
        return pd.DataFrame(results)

    def build_hist(self, df: pd.DataFrame, filename: str = "histograms.png"):
        """
        Строит вертикально расположенные гистограммы с наложенной кривой
        нормального распределения для всех числовых столбцов DataFrame
        и сохраняет результат в один файл.
        """
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        cols = df.select_dtypes(include=[np.number]).columns
        # Создаём фигуру с num_cols подграфиками в один столбец
        fig, axes = plt.subplots(num_cols, 1, figsize=(8, 4 * num_cols), constrained_layout=True)
        if num_cols == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            data = df[col].dropna()
            sns.histplot(data, kde=False, stat='density', ax=ax)
            mu, std = norm.fit(data)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, linewidth=2)
            ax.set_title(col)
        fig.savefig(filename)
        plt.close(fig)

    def run_test(self, df: pd.DataFrame):

        print("Тест случайности Run test")

        for col in df.columns:
            data = df[col]
            # Преобразуем данные в бинарную последовательность
            binary_data = (data > 0).astype(int)

            # Выполним тест на случайность
            runs_test = runstest_1samp(binary_data)
            print(col, "Тест на случайность:", runs_test)

    def check_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Тест на случайность методом серий относительно медианы.
        Возвращает DataFrame с колонками:
          column, median, num_runs, expected_runs, conclusion
        """
        results = []
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if s.empty:
                continue

            m = s.median()
            runs = np.where(s > m, 1, np.where(s < m, -1, 0))
            runs = runs[runs != 0]
            # число серий — сколько раз знак меняется +1
            num_runs = int(np.sum(np.diff(runs) != 0) + 1)
            # ожидаемое число серий для случайного ряда
            expected_runs = (2 * len(runs) - 1) / 3
            conclusion = ('случайны'
                          if abs(num_runs - expected_runs) <= 0.1 * expected_runs
                          else 'не случайны')

            results.append({
                'column': col,
                'median': float(m),
                'num_runs': num_runs,
                'expected_runs': round(expected_runs, 2),
                'conclusion': conclusion
            })

        return pd.DataFrame(results)

    def autocorr(self, df: pd.DataFrame):
        print("Тест случайности Автокорреляции")

        # Настройка графиков
        num_columns = len(df.columns)
        nrows = (num_columns + 1) // 2  # Определяем количество строк для подграфиков
        ncols = 2  # Количество столбцов, можно сделать больше, если нужно

        plt.figure(figsize=(15, 5 * nrows))  # Размер фигуры, чтобы она была достаточно большой

        # Строим графики автокорреляции для каждого столбца
        for i, column in enumerate(df.columns, 1):
            plt.subplot(nrows, ncols, i)  # Разбиваем на несколько подграфиков
            plot_acf(df[column], lags=20, ax=plt.gca())  # Строим коррелограмму для каждого столбца
            plt.title(f'Автокорреляция для {column}')

        plt.tight_layout()  # Подгоняем расположение графиков
        plt.show()

    def check_normal(self, df: pd.DataFrame):
        for column in df.columns:
            stat, p_value = shapiro(df[column])
            print(f'{column} - p-value: {p_value}')
            if p_value > 0.05:
                print(f'{column}: Данные можно считать нормальными (p > 0.05)')
            else:
                print(f'{column}: Данные не нормальны (p < 0.05)')

    def check_normal2(self, df: pd.DataFrame):
        for column in df.columns:
            result = anderson(df[column], dist='norm')
            print(f'{column} - Статистика: {result.statistic}, p-value: {result.significance_level}')
            if result.statistic < result.critical_values[2]:
                print(f'{column}: Данные можно считать нормальными')
            else:
                print(f'{column}: Данные не нормальны')
