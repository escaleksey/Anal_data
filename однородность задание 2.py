import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# 1. Чтение данных
df = pd.read_csv("kr2.csv")
X = df[['x1', 'x2', 'x3', 'x4', 'Y']]

# 2. Масштабирование (важно!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Метод K-средних (например, 3 кластера)
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Посмотрим, сколько объектов в каждом кластере
print("KMeans:")
print(df['Cluster_KMeans'].value_counts())

# 4. Метод матрицы расстояний (иерархическая кластеризация)
linked = linkage(X_scaled, method='ward')

# Рисуем дендрограмму
plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.)
plt.title('Дендрограмма')
plt.xlabel('Объекты')
plt.ylabel('Расстояние')
plt.grid()
plt.show()

# Можно задать число кластеров, например 3
df['Cluster_Hierarchical'] = fcluster(linked, t=3, criterion='maxclust')

print("\nИерархическая кластеризация:")
print(df['Cluster_Hierarchical'].value_counts())
