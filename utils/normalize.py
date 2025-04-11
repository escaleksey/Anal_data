import numpy as np
from scipy import stats
import pandas as pd

from sklearn.preprocessing import PowerTransformer


class Normalizer:
    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            # Проверяем, есть ли нули или отрицательные значения, добавляем 1, если нужно
            if (df[column] <= 0).any():
                df[column] = df[column] + np.abs(df[column].min()) + 1  # Добавляем необходимую константу

            # Применяем Box-Cox трансформацию
            df[column], _ = stats.boxcox(df[column])
        return df

    @staticmethod
    def normalize_yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
        transformer = PowerTransformer(method='yeo-johnson')
        df_transformed = pd.DataFrame(transformer.fit_transform(df), columns=df.columns)
        return df_transformed

    @staticmethod
    def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            if (df[column] <= 0).any():
                df[column] = df[column] + np.abs(
                    df[column].min()) + 1  # Добавляем константу, если есть нули или отрицательные значения
            df[column] = np.log(df[column])  # Логарифмическая трансформация
        return df
