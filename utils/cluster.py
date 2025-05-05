def k_means_clustering(df, k=3, max_iter=100):
    import numpy as np
    from matplotlib import pyplot as plt

    # Масштабируем все столбцы (X1, X2, X3, X4, Y)
    columns = ['x1', 'x2', 'x3', 'x4', 'Y']
    data = df[columns].values.astype(float)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data = (data - data_min) / (data_max - data_min)

    # Этап 1: задаём начальные центроиды (случайные 3 строки)
    np.random.seed(42)
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices]

    clusters = np.zeros(len(data))

    for iteration in range(max_iter):

        # Этап 2: считаем расстояния до центроидов
        distances = np.zeros((len(data), k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centroids[i], axis=1)

        # Этап 3: присваиваем каждой точке ближайший кластер
        new_clusters = np.argmin(distances, axis=1)

        # Проверяем, изменились ли кластеры
        if np.array_equal(clusters, new_clusters):
            print(f"Алгоритм завершился на итерации {iteration+1}")
            break

        clusters = new_clusters

        # Этап 4: пересчитываем центроиды
        for i in range(k):
            points = data[clusters == i]
            if len(points) > 0:
                centroids[i] = points.mean(axis=0)

    # Выводим количество записей и средние значения (в исходных масштабах)
    for i in range(k):
        points = data[clusters == i]
        count = len(points)
        mean_scaled = points.mean(axis=0)

        # Обратная нормализация
        mean_original = mean_scaled * (data_max - data_min) + data_min

        # Округляем и превращаем в обычные float (чтобы не было np.float)
        mean_original_rounded = np.round(mean_original, 3)
        mean_dict = {col: float(val) for col, val in zip(columns, mean_original_rounded)}

        print(f"Кластер {i + 1}: {count} записей")
        print(f"Средние значения признаков (до масштабирования): {mean_dict}\n")