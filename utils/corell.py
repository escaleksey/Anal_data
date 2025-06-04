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
    def build_regression_models(df, significance_matrix, y_col):
        numeric_df = df.select_dtypes(include=[np.number])
        cols = numeric_df.columns
        if y_col not in cols:
            print(f"Целевой признак {y_col} отсутствует в данных.")
            return {}

        models = {}

        for i in range(len(cols)):
            x_col = cols[i]
            if x_col == y_col:
                continue  # пропускаем целевой признак в качестве X

            # Проверяем значимость корреляции между x_col и y_col
            if significance_matrix.at[x_col, y_col]:
                x = numeric_df[x_col]
                y = numeric_df[y_col]

                n = len(x)
                m = 2  # параметры: k и b

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

                    RSS = np.sum((y - y_pred) ** 2)
                    TSS = np.sum((y - y_mean) ** 2)
                    ESS = np.sum((y_pred - y_mean) ** 2)

                    sigma_ost = np.sqrt(RSS / (n - m))
                    sigma_y = np.std(y, ddof=1)
                    eta2 = 1 - (sigma_ost ** 2) / (sigma_y ** 2)

                    models[x_col] = {
                        'k': k,
                        'b': b,
                        'σ_ост': sigma_ost,
                        'η²': eta2,
                        'TSS': TSS,
                        'ESS': ESS,
                        'RSS': RSS
                    }

                    print(f"Регрессия для {y_col} = k * {x_col} + b:")
                    print(f"  k = {k:.4f}, b = {b:.4f}")
                    print(f"  σ_ост = {sigma_ost:.4f}, η² = {eta2:.4f}")
                    print(f"  TSS = {TSS:.4f}, ESS = {ESS:.4f}, RSS = {RSS:.4f}")
                    print("  Проверка: TSS ≈ ESS + RSS →",
                          f"{TSS:.4f} ≈ {ESS + RSS:.4f}")
                    print("-" * 60)

                except np.linalg.LinAlgError:
                    print(f"Система вырождена для пары {x_col} и {y_col}")

        return models

    @staticmethod
    def build_multiple_regression_model(df, x1_col, x2_col, y_col, verbose=True):
        import numpy as np

        x1 = df[x1_col].astype(float).values
        x2 = df[x2_col].astype(float).values
        y = df[y_col].astype(float).values
        n = len(y)
        m = 3  # k1, k2, b

        # Матрица признаков X с добавлением единиц для свободного члена
        X = np.column_stack((x1, x2, np.ones(n)))

        # Метод наименьших квадратов: b = (X^T X)^-1 X^T y
        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            k1, k2, b = coef
            y_pred = X @ coef
            y_mean = np.mean(y)

            # Метрики
            RSS = np.sum((y - y_pred) ** 2)
            TSS = np.sum((y - y_mean) ** 2)
            ESS = np.sum((y_pred - y_mean) ** 2)

            sigma_ost = np.sqrt(RSS / (n - m))
            sigma_y = np.std(y, ddof=1)
            eta2 = 1 - (sigma_ost ** 2) / (sigma_y ** 2)

            if verbose:
                print(f"Модель: {y_col} = {k1:.4f}*{x1_col} + {k2:.4f}*{x2_col} + {b:.4f}")
                print(f"  σ_ост = {sigma_ost:.4f}, η² = {eta2:.4f}")
                print(f"  TSS = {TSS:.4f}, ESS = {ESS:.4f}, RSS = {RSS:.4f}")
                print(f"  Проверка: TSS ≈ ESS + RSS → {TSS:.4f} ≈ {(ESS + RSS):.4f}")
                print("-" * 60)

            return {
                'k1': k1,
                'k2': k2,
                'b': b,
                'σ_ост': sigma_ost,
                'η²': eta2,
                'TSS': TSS,
                'ESS': ESS,
                'RSS': RSS
            }

        except np.linalg.LinAlgError:
            print("Ошибка: Система вырождена или плохо обусловлена.")
            return None

    @staticmethod
    def build_multiple_regression_model_3x(df, x1_col, x2_col, x3_col, y_col, verbose=True):
        """
        Строит множественную линейную регрессию по трём признакам:
        y = k1*x1 + k2*x2 + k3*x3 + b

        :param df: DataFrame с данными
        :param x1_col, x2_col, x3_col: имена колонок с признаками
        :param y_col: имя колонки с зависимой переменной
        :param verbose: выводить подробный отчет в консоль
        :return: dict с коэффициентами и метриками или None при ошибке
        """
        x1 = df[x1_col].astype(float).values
        x2 = df[x2_col].astype(float).values
        x3 = df[x3_col].astype(float).values
        y = df[y_col].astype(float).values

        n = len(y)
        m = 4  # 3 коэффициента + свободный член

        # Формируем матрицу X с добавленным столбцом единиц для свободного члена
        X = np.column_stack((x1, x2, x3, np.ones(n)))

        try:
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            k1, k2, k3, b = coef

            y_pred = X @ coef
            y_mean = np.mean(y)

            RSS = np.sum((y - y_pred) ** 2)  # остаточная сумма квадратов
            TSS = np.sum((y - y_mean) ** 2)  # общая сумма квадратов
            ESS = np.sum((y_pred - y_mean) ** 2)  # объяснённая сумма квадратов

            sigma_ost = np.sqrt(RSS / (n - m))  # среднеквадратичная ошибка остатков
            sigma_y = np.std(y, ddof=1)
            eta2 = 1 - (sigma_ost ** 2) / (sigma_y ** 2)  # коэффициент детерминации по остатку

            if verbose:
                print(f"Модель: {y_col} = {k1:.4f}*{x1_col} + {k2:.4f}*{x2_col} + {k3:.4f}*{x3_col} + {b:.4f}")
                print(f"  σ_ост = {sigma_ost:.4f}, η² = {eta2:.4f}")
                print(f"  TSS = {TSS:.4f}, ESS = {ESS:.4f}, RSS = {RSS:.4f}")
                print(f"  Проверка: TSS ≈ ESS + RSS → {TSS:.4f} ≈ {(ESS + RSS):.4f}")
                print("-" * 60)

            return {
                'k1': k1,
                'k2': k2,
                'k3': k3,
                'b': b,
                'σ_ост': sigma_ost,
                'η²': eta2,
                'TSS': TSS,
                'ESS': ESS,
                'RSS': RSS
            }

        except np.linalg.LinAlgError:
            print("Ошибка: Система вырождена или плохо обусловлена.")
            return None


    @staticmethod
    def select_features_for_y(df, y_col, significance_matrix):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [col for col in numeric_cols if col != y_col and significance_matrix.at[col, y_col]]

        if not candidate_features:
            print(f"Нет значимых признаков для {y_col}")
            return ()

        # Ищем пару признаков, которые значимо коррелируют с y_col,
        # но при этом между собой значимость корреляции отсутствует (False)
        for i in range(len(candidate_features)):
            for j in range(i + 1, len(candidate_features)):
                f1, f2 = candidate_features[i], candidate_features[j]
                if not significance_matrix.at[f1, f2]:
                    print(
                        f"Найдена пара признаков для {y_col}: {f1}, {f2} (между ними значимость корреляции отсутствует)")
                    return (f1, f2)

        # Если пары не найдено — берем один значимый признак
        print(
            f"Нет пар признаков с отсутствием значимой корреляции между собой для {y_col}, берём один признак: {candidate_features[0]}")
        return (candidate_features[0],)
