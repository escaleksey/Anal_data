import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'
from matplotlib import pyplot as plt
import numpy as np
from utils import CheckOutliers, CheckLength, CheckRandom, Normalizer, k_means_clustering, CheckCorell


from scipy.stats import chi2, norm, expon

# фиксируем сид для воспроизводимости
np.random.seed(42)

# количество строк
n = 500

# создаём DataFrame
df = pd.read_csv("kr.csv")

 # Печать уникальных значений в Y
co = CheckOutliers()
cl = CheckLength()
# co.check_sigmas(df)
# co.check_quantiles(df)
# co.check_grabbs(df)

cr = CheckRandom()

#res = CheckCorell().check_correlation(df)
#res2 = CheckCorell().build_regression_models(df, res[-1])
from create_word import create

create(df)



