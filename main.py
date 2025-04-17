import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
from matplotlib import pyplot as plt
import numpy as np
from utils import CheckOutliers, CheckLength, CheckRandom, Normalizer


from scipy.stats import chi2, norm, expon
column = "X4"

# фиксируем сид для воспроизводимости
np.random.seed(42)

# количество строк
n = 500

# создаём DataFrame
df = pd.DataFrame({
    'x1': np.random.normal(loc=50, scale=4, size=n),
    'x2': np.random.normal(loc=100.5, scale=27, size=n),
    'x3': np.random.normal(loc=-12.4, scale=14, size=n),
    'x4': np.random.normal(loc=23, scale=6, size=n),
    'Y':  np.random.normal(loc=230, scale=15, size=n),
})
 # Печать уникальных значений в Y
co = CheckOutliers()
cl = CheckLength()
# co.check_sigmas(df)
# co.check_quantiles(df)
# co.check_grabbs(df)

cr = CheckRandom()
#cr.check_median(df)

cr.check_pirs(df)
#cr.run_test(df)
#cr.autocorr(df)

#cr.check_median(df)
#
# print("До нормализации")
# cr.check_normal(df)
# df = Normalizer.normalize_log(df)
# print("После нормализации")
# cr.check_normal(df)



