import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# 1. Чтение данных
df = pd.read_csv("kr.csv")
X = df[['x1', 'x2', 'x3', 'x4', 'Y']]

# 2. Масштабирование (важно!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Метод K-средних (например, 3 кластера)
kmeans = KMeans(random_state=0, n_init=10)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Посмотрим, сколько объектов в каждом кластере
print("KMeans:")
print(df['Cluster_KMeans'].value_counts())

# 4. Метод матрицы расстояний (иерархическая кластеризация)
linked = linkage(X_scaled, method='ward')

# Можно задать число кластеров, например 3
df['Cluster_Hierarchical'] = fcluster(linked, t=8, criterion='maxclust')

print("\nИерархическая кластеризация:")
print(df['Cluster_Hierarchical'].value_counts())
