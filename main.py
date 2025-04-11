import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'

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

# print(cl.ci_mean(df))
# cl.plot_mean_stability(df)

cr = CheckRandom()
print("До нормализации")
cr.check_normal(df)
df = Normalizer.normalize_log(df)
print("После нормализации")
cr.check_normal(df)



