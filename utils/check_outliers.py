import pandas as pd
from scipy import stats
import numpy as np

class CheckOutliers():
    def check_quantiles(self, df: pd.DataFrame):
        print("Quantiles calculated")
        outliers_dict = {}

        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            print(f"{col}: Q1 - {round(Q1, 3)}; Q3 - {round(Q3, 3)}; "
                  f"{(float(df[col].min()), round(lower))},"
                  f" {(float(df[col].max()), round(upper))}")
            outliers = df[(df[col] < lower) | (df[col] > upper)]

            if not outliers.empty:
                outliers_dict[col] = outliers

        # Вывод выбросов по столбцам
        for col, out_df in outliers_dict.items():
            print(f"\nВыбросы в колонке '{col}':")
            print(out_df[[col]])

    def check_sigmas(self, df: pd.DataFrame):
        print("STD (3 сигмы) calculated")
        outliers_dict = {}
        for col in df.columns:
            avg = float(df[col].mean())
            std = float(df[col].std())
            print(f"{col}: avg - {round(avg, 3)}; std - {round(std, 3)}; "
                  f"{(float(df[col].min()),  round(avg-3*std), 3)},"
                  f" {(float(df[col].max()), round(avg+3*std))}")
            outliers = df[(df[col] < avg-3*std) | (df[col] > avg+3*std)]

            if not outliers.empty:
                outliers_dict[col] = outliers

                # Вывод выбросов по столбцам
        for col, out_df in outliers_dict.items():
            print(f"\nВыбросы в колонке '{col}':")
            print(out_df[[col]])

    def check_grabbs(self, df: pd.DataFrame, alpha=0.05):
        print("Grubbs test calculated")
        print("⚠ WARNING! Проверьте, что у данных нормальное распределение\n")

        outliers_dict = {}

        for col in df.columns:
            series = df[col].dropna()
            if len(series) < 3:
                print(f"{col}: слишком мало данных для теста Граббса\n")
                continue

            mean = series.mean()
            std = series.std()
            n = len(series)
            max_deviation = abs(series - mean).max()
            max_val = series[abs(series - mean) == max_deviation].iloc[0]
            G = max_deviation / std
            t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t ** 2 / (n - 2 + t ** 2))

            print(f"{col}: avg = {round(mean, 3)}, std = {round(std, 3)}")
            print(
                f"G = {round(G, 3)}, G_crit = {round(G_crit, 3)} → {'выброс найден' if G > G_crit else 'выброс не обнаружен'}")

            if G > G_crit:
                outliers = df[df[col] == max_val]
                outliers_dict[col] = outliers

        for col, out_df in outliers_dict.items():
            print(f"\nВыброс в колонке '{col}':")
            print(out_df[[col]])