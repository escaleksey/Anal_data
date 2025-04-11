import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
from matplotlib import pyplot as plt

from utils import CheckOutliers, CheckLength, CheckRandom, Normalizer


column = "X4"
df = pd.read_csv("kr.csv", delimiter=";")
df = df.iloc[:, 1:]
 # Печать уникальных значений в Y
co = CheckOutliers()
cl = CheckLength()
# co.check_sigmas(df)
# co.check_quantiles(df)
# co.check_grabbs(df)

cr = CheckRandom()
#cr.check_median(df)

#cr.check_pirs(df)
#cr.run_test(df)
#cr.autocorr(df)

df['Y_MA'] = df['Y'].rolling(window=10, min_periods=1).mean()
step = 1  # каждый пятый элемент, например
plt.figure(figsize=(12, 6))
plt.plot(df.index[::step], df['Y'][::step], label='Оригинальные данные')
plt.plot(df.index[::step], df['Y_MA'][::step], color='red', label='Скользящее среднее')
plt.title("Скользящее среднее для Y")
plt.legend()
plt.show()

#
# print("До нормализации")
# cr.check_normal(df)
# df = Normalizer.normalize_log(df)
# print("После нормализации")
# cr.check_normal(df)



