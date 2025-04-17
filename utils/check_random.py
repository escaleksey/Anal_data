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
                   bins: int = 8,
                   param_estimator=None):
        """
        Применяет χ²-тест согласия Пирсона ко всем числовым столбцам DataFrame.

        Параметры:
        - df: pd.DataFrame — таблица с числовыми данными.
        - dist — распределение из scipy.stats (по умолчанию: нормальное);
        - bins: int — количество интервалов;
        - param_estimator — функция: pd.Series → (loc, scale).
            Если None — используется (mean, std).

        Возвращает: DataFrame с результатами по столбцам.
        """
        if param_estimator is None:
            param_estimator = lambda s: (s.mean(), s.std(ddof=0))

        results = []

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            n = len(series)
            if n < bins:
                print(f"⚠️ Пропущено: {col} — слишком мало данных для {bins} бинов.")
                continue

            params = param_estimator(series)
            x = series.values

            # Границы интервалов по квантилям
            probs = np.linspace(0, 1, bins + 1)
            edges = dist.ppf(probs, *params)
            edges[0] = min(edges[0], x.min() - 1e-6)
            edges[-1] = max(edges[-1], x.max() + 1e-6)

            obs_counts, _ = np.histogram(x, bins=edges)
            cdf_vals = dist.cdf(edges, *params)
            exp_probs = np.diff(cdf_vals)
            exp_counts = n * exp_probs

            # χ²-статистика
            chi2_stat = ((obs_counts - exp_counts) ** 2 / exp_counts).sum()
            dfree = bins - 1 - len(params)
            p_value = 1 - chi2.cdf(chi2_stat, dfree)

            if np.any(exp_counts < 5):
                note = "E_i < 5"
            else:
                note = ""

            results.append({
                'column': col,
                'chi2_stat': round(chi2_stat, 3),
                'p_value': round(p_value, 4),
                'df': dfree,
                'note': note
            })

        print(results)

    def build_hist(self, df: pd.DataFrame):
        # Настройка графиков
        # Настройка графиков
        plt.figure(figsize=(15, 5))

        # Проходим по каждому столбцу DataFrame и строим гистограмму и график нормального распределения
        for i, column in enumerate(df.columns, 1):
            plt.subplot(1, len(df.columns), i)  # Разбиваем на несколько подграфиков
            sns.histplot(df[column], kde=True, stat='density')  # Строим гистограмму с KDE (нормированная плотность)

            # Вычисляем параметры нормального распределения для данных
            mu, std = norm.fit(df[column])

            # Строим график нормального распределения
            xmin, xmax = plt.xlim()  # Получаем пределы оси x
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)  # Кривая нормального распределения

            # Добавляем заголовок
            plt.title(f'Гистограмма {column}')

        plt.tight_layout()  # Подгоняем расположение графиков
        plt.show()

    def run_test(self, df: pd.DataFrame):

        print("Тест случайности Run test")

        for col in df.columns:
            data = df[col]
            # Преобразуем данные в бинарную последовательность
            binary_data = (data > 0).astype(int)

            # Выполним тест на случайность
            runs_test = runstest_1samp(binary_data)
            print(col, "Тест на случайность:", runs_test)

    def check_median(self, df: pd.DataFrame):
        """"СТАНДАРТНЫЙ МЕТОД КОТОРЫЙ МЫ УЧИЛИ НА ПАРАХ"""
        for column in df.columns:
            print(f"Тест методом серий относительно медиан для {column}")

            # Вычисление медианы для столбца
            median = df[column].median()

            # Преобразуем данные: выше медианы -> 1, ниже медианы -> -1
            series = np.where(df[column] > median, 1, np.where(df[column] < median, -1, 0))

            # Пропускаем нули (равные медиане)
            series = series[series != 0]

            # Подсчитываем количество серий (смена знака)
            num_series = np.sum(np.diff(series) != 0)

            print(f"Медиана для {column}: {median}")
            print(f"Количество серий для {column}: {num_series}")

            # В данном случае, чтобы проверить случайность, можно сравнить количество серий с ожидаемым
            # Примерное ожидаемое количество серий для случайного процесса
            expected_num_series = len(series) / 2
            print(f"Ожидаемое количество серий: {expected_num_series}")

            if num_series > expected_num_series * 0.9 and num_series < expected_num_series * 1.1:
                print(f"{column}: Данные можно считать случайными (количество серий близко к ожидаемому)")
            else:
                print(f"{column}: Данные не случайны (количество серий значительно отличается от ожидаемого)")

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
