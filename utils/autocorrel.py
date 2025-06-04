import pandas as pd
import numpy as np
from scipy.stats import pearsonr, t

class Autocorrel:

    @staticmethod
    def check_lags(df: pd.DataFrame, lag_count: int, y_col: str = "Y", alpha: float = 0.05):
        """
        Проверяет автокорреляцию Y с лагами каждой переменной X.
        Возвращает три таблицы: корреляции, R² и значимость.

        :param df: DataFrame с числовыми колонками
        :param lag_count: максимальное число лагов
        :param y_col: колонка зависимой переменной
        :param alpha: уровень значимости для t-критерия
        :return: tuple(DataFrame, DataFrame, DataFrame) — корреляции, R², значимость
        """
        numeric_df = df.select_dtypes(include=[np.number])
        x_cols = [col for col in numeric_df.columns if col != y_col]

        correlation_df = pd.DataFrame(index=x_cols, columns=[f"Lag {i}" for i in range(1, lag_count + 1)])
        r2_df = pd.DataFrame(index=x_cols, columns=[f"Lag {i}" for i in range(1, lag_count + 1)])
        significance_df = pd.DataFrame(index=x_cols, columns=[f"Lag {i}" for i in range(1, lag_count + 1)])

        for x in x_cols:
            for lag in range(1, lag_count + 1):
                shifted_x = numeric_df[x].shift(lag)
                y = numeric_df[y_col]
                valid_idx = (~shifted_x.isna()) & (~y.isna())

                if valid_idx.sum() > 2:
                    sample_size = valid_idx.sum()
                    x_lag = shifted_x[valid_idx]
                    y_valid = y[valid_idx]

                    r, _ = pearsonr(x_lag, y_valid)
                    r2 = r ** 2

                    # t-статистика и проверка значимости
                    if abs(r) < 1:
                        t_stat = abs(r) / np.sqrt(1 - r ** 2) * np.sqrt(sample_size - 2)
                    else:
                        t_stat = np.inf
                    t_critical = t.ppf(1 - alpha / 2, df=sample_size - 2)
                    significant = t_stat > t_critical

                    correlation_df.loc[x, f"Lag {lag}"] = round(r, 4)
                    r2_df.loc[x, f"Lag {lag}"] = round(r2, 4)
                    significance_df.loc[x, f"Lag {lag}"] = significant
                else:
                    correlation_df.loc[x, f"Lag {lag}"] = np.nan
                    r2_df.loc[x, f"Lag {lag}"] = np.nan
                    significance_df.loc[x, f"Lag {lag}"] = False

        return correlation_df, r2_df, significance_df

    @staticmethod
    def shift_by_significant_lags(df: pd.DataFrame,
                                  corr_df: pd.DataFrame,
                                  significance_df: pd.DataFrame,
                                  y_col: str = "Y") -> pd.DataFrame:
        """
        Сдвигает X на их значимые лаги, удаляет незначимые X,
        и обрезает строки сверху/снизу по соответствующим правилам.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        x_cols = [col for col in numeric_df.columns if col != y_col]

        lag_map = {}  # {x: lag}
        max_lag = 0

        for x in x_cols:
            for lag_num, lag_label in enumerate(corr_df.columns, start=1):
                if significance_df.loc[x, lag_label]:
                    lag_map[x] = lag_num
                    max_lag = max(max_lag, lag_num)
                    break  # берем первый значимый

        # Убираем X без значимого лага
        if not lag_map:
            raise ValueError("Нет значимых лагов для ни одной переменной")

        # Сдвигаем X по своим лагам
        shifted_data = {}
        for x, lag in lag_map.items():
            shifted_data[x] = df[x].shift(lag)

        # Оставляем Y без сдвига, но обрежем его сверху потом
        shifted_data[y_col] = df[y_col]

        result_df = pd.DataFrame(shifted_data)

        # Вычисляем сколько строк обрезать:
        # - Сверху: максимальный лаг (чтобы у Y и всех X были соответствующие значения)
        # - Снизу: для каждого X удаляем свои последние lag строк (они стали NaN после shift)
        top_cut = max_lag
        bottom_cuts = [lag for lag in lag_map.values()]
        bottom_cut = max(bottom_cuts) if bottom_cuts else 0

        # Удаляем строки
        result_df = result_df.iloc[top_cut:len(result_df) - bottom_cut].reset_index(drop=True)

        return result_df


