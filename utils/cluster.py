def k_means_clustering(df, k_values=[2, 3, 4, 5], max_iter=100, save_path='clustering_metrics.png'):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score

    # Масштабируем
    columns = ['x1', 'x2', 'x3', 'x4', 'Y']
    data = df[columns].values.astype(float)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_scaled = (data - data_min) / (data_max - data_min)

    inertias = []
    silhouettes = []
    report_data = []

    for k in k_values:
        np.random.seed(42)
        indices = np.random.choice(len(data_scaled), k, replace=False)
        centroids = data_scaled[indices]

        clusters = np.zeros(len(data_scaled))

        for iteration in range(max_iter):
            distances = np.zeros((len(data_scaled), k))
            for i in range(k):
                distances[:, i] = np.linalg.norm(data_scaled - centroids[i], axis=1)

            new_clusters = np.argmin(distances, axis=1)

            if np.array_equal(clusters, new_clusters):
                break

            clusters = new_clusters

            for i in range(k):
                points = data_scaled[clusters == i]
                if len(points) > 0:
                    centroids[i] = points.mean(axis=0)

        # Inertia
        inertia = sum(np.sum((data_scaled[clusters == i] - centroids[i]) ** 2) for i in range(k))
        inertias.append(inertia)

        # Silhouette
        silhouette = silhouette_score(data_scaled, clusters) if k > 1 else float('nan')
        silhouettes.append(silhouette)

        # Собираем данные по кластерам
        cluster_info = []
        for i in range(k):
            points = data_scaled[clusters == i]
            count = len(points)
            mean_scaled = points.mean(axis=0)
            mean_original = mean_scaled * (data_max - data_min) + data_min
            mean_rounded = np.round(mean_original, 3)
            mean_dict = {col: float(val) for col, val in zip(columns, mean_rounded)}

            cluster_info.append({
                'cluster': i + 1,
                'count': int(count),
                'mean_values': mean_dict
            })

        report_data.append({
            'k': k,
            'inertia': inertia,
            'silhouette': silhouette,
            'clusters': cluster_info
        })

    # Графики
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Inertia')
    plt.title('Метод локтя')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouettes, 'go-')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Коэффициент силуэта')

    plt.tight_layout()
    plt.savefig(save_path)  # Сохраняем как картинку
    plt.close()

    return report_data


def assign_clusters(df, k, max_iter=100):
    import numpy as np

    columns = ['x1', 'x2', 'x3', 'x4', 'Y']
    data = df[columns].values.astype(float)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_scaled = (data - data_min) / (data_max - data_min)

    np.random.seed(42)
    indices = np.random.choice(len(data_scaled), k, replace=False)
    centroids = data_scaled[indices]
    clusters = np.zeros(len(data_scaled))

    for iteration in range(max_iter):
        distances = np.zeros((len(data_scaled), k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(data_scaled - centroids[i], axis=1)

        new_clusters = np.argmin(distances, axis=1)
        if np.array_equal(clusters, new_clusters):
            break

        clusters = new_clusters
        for i in range(k):
            points = data_scaled[clusters == i]
            if len(points) > 0:
                centroids[i] = points.mean(axis=0)

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters.astype(int)
    return df_with_clusters

