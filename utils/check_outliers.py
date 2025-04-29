import pandas as pd
from scipy import stats
import numpy as np

class CheckOutliers():
    def check_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверка выбросов по межквартильному размаху (IQR).
        Возвращает DataFrame с колонками:
          column, num_outliers, percent_outliers, outlier_values,
          lower_bound, upper_bound
        """
        results = []
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            percent = 100 * len(outliers) / len(series)

            outlier_values = ', '.join(map(lambda x: str(round(x, 3)), outliers[:20]))
            if len(outliers) > 20:
                outlier_values += ", ..."

            results.append({
                'column': col,
                'num_outliers': len(outliers),
                'percent_outliers': round(percent, 2),
                'outlier_values': outlier_values,
                'lower_bound': round(float(lower), 3),
                'upper_bound': round(float(upper), 3),
            })
        return pd.DataFrame(results)

    def check_sigmas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Проверка выбросов по правилу 3 сигм (среднее ± 3*ст. отклонение).
        Возвращает DataFrame с колонками:
          column, mean, std, lower_bound, upper_bound, num_outliers, percent_outliers, outlier_values
        """
        print("STD (3 сигмы) calculated")
        results = []

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            avg = float(series.mean())
            std = float(series.std())
            lower = avg - 3 * std
            upper = avg + 3 * std

            outliers = series[(series < lower) | (series > upper)]
            percent = 100 * len(outliers) / len(series)

            outlier_values = ', '.join(map(lambda x: str(round(x, 3)), outliers[:20]))
            if len(outliers) > 20:
                outlier_values += ", ..."

            results.append({
                'column': col,
                'mean': round(avg, 3),
                'std': round(std, 3),
                'lower_bound': round(lower, 3),
                'upper_bound': round(upper, 3),
                'num_outliers': len(outliers),
                'percent_outliers': round(percent, 2),
                'outlier_values': outlier_values
            })

        return pd.DataFrame(results)

    def check_grabbs(self, df: pd.DataFrame, alpha=0.05) -> pd.DataFrame:
        """
        Тест Граббса на выбросы. Проверяет наличие одного выброса в выборке.
        Возвращает DataFrame с колонками:
          column, mean, std, G, G_crit, outlier_value, conclusion
        """
        print("Grubbs test calculated")
        print("⚠ WARNING! Проверьте, что у данных нормальное распределение\n")

        results = []

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 3:
                print(f"{col}: слишком мало данных для теста Граббса\n")
                continue

            mean = series.mean()
            std = series.std()
            n = len(series)
            deviations = abs(series - mean)
            max_deviation = deviations.max()
            max_val = series[deviations == max_deviation].iloc[0]
            G = max_deviation / std
            t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t ** 2 / (n - 2 + t ** 2))
            is_outlier = G > G_crit

            print(f"{col}: avg = {round(mean, 3)}, std = {round(std, 3)}")
            print(
                f"G = {round(G, 3)}, G_crit = {round(G_crit, 3)} → {'выброс найден' if is_outlier else 'выброс не обнаружен'}")

            results.append({
                'column': col,
                'mean': round(mean, 3),
                'std': round(std, 3),
                'G': round(G, 3),
                'G_crit': round(G_crit, 3),
                'outlier_value': round(max_val, 3) if is_outlier else '',
                'conclusion': 'выброс найден' if is_outlier else 'выброс не обнаружен'
            })

        return pd.DataFrame(results)
