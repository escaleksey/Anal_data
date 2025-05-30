import pandas as pd
from scipy import stats
import numpy as np
from scipy.stats import t

class CheckCorell():
    @staticmethod
    def check_correlation(df, alpha=0.05):
        numeric_df = df.select_dtypes(include=[np.number])
        cols = numeric_df.columns
        n = len(cols)
        sample_size = len(numeric_df)

        corr_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
        r2_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
        t_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
        significance_matrix = pd.DataFrame(np.full((n, n), False), columns=cols, index=cols)

        for i in range(n):
            for j in range(n):
                x = numeric_df[cols[i]]
                y = numeric_df[cols[j]]

                x_mean = x.mean()
                y_mean = y.mean()

                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
                r = numerator / denominator if denominator != 0 else 0

                # t-статистика
                t_op = abs(r) / np.sqrt(1 - r ** 2) * np.sqrt(sample_size - 2) if abs(r) < 1 else np.inf
                t_critical = t.ppf(1 - alpha / 2, df=sample_size - 2)
                significant = t_op > t_critical

                corr_matrix.iloc[i, j] = r
                r2_matrix.iloc[i, j] = r ** 2
                t_matrix.iloc[i, j] = t_op
                significance_matrix.iloc[i, j] = significant

        print("Матрица коэффициентов корреляции:")
        print(corr_matrix.round(3))
        print("\nМатрица детерминации (R²):")
        print(r2_matrix.round(3))
        print("\nМатрица t-статистик:")
        print(t_matrix.round(3))
        print(f"\nМатрица значимости (t_op > t_critical при alpha = {alpha}):")
        print(significance_matrix)

        return corr_matrix, r2_matrix, t_matrix, significance_matrix

    @staticmethod
    def build_regression_models(df, significance_matrix):
        numeric_df = df.select_dtypes(include=[np.number])
        cols = numeric_df.columns
        models = {}

        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j and significance_matrix.iloc[i, j]:
                    x = numeric_df[cols[i]]
                    y = numeric_df[cols[j]]

                    n = len(x)
                    m = 2  # число параметров: k и b

                    Sx = np.sum(x)
                    Sy = np.sum(y)
                    Sxx = np.sum(x ** 2)
                    Sxy = np.sum(x * y)

                    A = np.array([[Sxx, Sx], [Sx, n]])
                    B = np.array([Sxy, Sy])

                    try:
                        k, b = np.linalg.solve(A, B)
                        y_pred = k * x + b
                        y_mean = np.mean(y)

                        # Качество модели
                        RSS = np.sum((y - y_pred) ** 2)
                        TSS = np.sum((y - y_mean) ** 2)
                        ESS = np.sum((y_pred - y_mean) ** 2)

                        sigma_ost = np.sqrt(RSS / (n - m))
                        sigma_y = np.std(y, ddof=1)
                        eta2 = 1 - (sigma_ost ** 2) / (sigma_y ** 2)

                        models[(cols[i], cols[j])] = {
                            'k': k,
                            'b': b,
                            'σ_ост': sigma_ost,
                            'η²': eta2,
                            'TSS': TSS,
                            'ESS': ESS,
                            'RSS': RSS
                        }

                        print(f"Регрессия для {cols[j]} = k * {cols[i]} + b:")
                        print(f"  k = {k:.4f}, b = {b:.4f}")
                        print(f"  σ_ост = {sigma_ost:.4f}, η² = {eta2:.4f}")
                        print(f"  TSS = {TSS:.4f}, ESS = {ESS:.4f}, RSS = {RSS:.4f}")
                        print("  Проверка: TSS ≈ ESS + RSS →",
                              f"{TSS:.4f} ≈ {ESS + RSS:.4f}")
                        print("-" * 60)

                    except np.linalg.LinAlgError:
                        print(f"Система вырождена для пары {cols[i]} и {cols[j]}")

        return models