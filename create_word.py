from utils import CheckOutliers
from utils.check_random import CheckRandom
from docx import Document
from docx.shared import Inches
import pandas as pd
import numpy as np
from scipy.stats import chi2
from utils import CheckCorell

def export_pirs_report(pirs_res: pd.DataFrame, doc: Document):
    """
    Добавляет в doc главу с результатами χ²‑теста Пирсона.
    pirs_res: DataFrame с колонками ['column','chi2_stat','p_value','df','note'].
    """
    doc.add_heading("Глава 1. χ²‑тест Пирсона на нормальность", level=1)
    for row in pirs_res.to_dict('records'):
        doc.add_heading(f"Столбец: {row['column']}", level=2)
        doc.add_paragraph(f"χ²‑статистика: {row['chi2_stat']}")
        doc.add_paragraph(f"Степени свободы: {row['df']}")
        doc.add_paragraph(f"p‑значение: {row['p_value']}")
        if row['note']:
            doc.add_paragraph(f"⚠️ Примечание: {row['note']}", style='IntenseQuote')
        conclusion = ("Нет оснований отвергнуть нормальность."
                      if row['p_value'] >= 0.05 else
                      "Отвергаем нормальность.")
        doc.add_paragraph(f"Вывод: {conclusion}", style='Quote')
        doc.add_paragraph()


def export_median_report(median_res: pd.DataFrame, doc: Document):
    """
    Добавляет в doc главу с результатами теста серий относительно медианы.
    median_res: DataFrame с колонками ['column','median','num_runs','expected_runs','conclusion'].
    """
    doc.add_heading("Глава 2. Тест на случайность методом серий", level=1)
    for row in median_res.to_dict('records'):
        doc.add_heading(f"Столбец: {row['column']}", level=2)
        doc.add_paragraph(f"Медиана: {row['median']}")
        doc.add_paragraph(f"Число серий: {row['num_runs']}")
        doc.add_paragraph(f"Ожидаемое число серий: {row['expected_runs']}")

        doc.add_paragraph(f"Максимальная серия: {row['max_run_length']}")
        doc.add_paragraph(f"Ожидаемая серия: {row['b crit']}")
        doc.add_paragraph(f"Вывод: Данные {row['conclusion']}", style='Quote')
        doc.add_paragraph()


def export_histograms_report(doc: Document, filename: str = "histograms.png"):
    """
    Добавляет в doc главу с гистограммами, используя build_hist.
    """
    # Построение и сохранение общей гистограммы
    doc.add_heading("Глава 3. Гистограммы с нормальной кривой", level=1)
    doc.add_picture(filename, width=Inches(6))
    doc.add_paragraph()


def export_outliers_report(outliers_res: pd.DataFrame, doc: Document):
    """
    Добавляет в doc главу с анализом выбросов.
    """
    doc.add_heading("Глава 4. Анализ выбросов", level=1)
    for row in outliers_res.to_dict('records'):
        doc.add_heading(f"Столбец: {row['column']}", level=2)
        doc.add_paragraph(f"Минимально допустимое значение: {row['lower_bound']}")
        doc.add_paragraph(f"Максимально допустимое значение: {row['upper_bound']}")
        doc.add_paragraph(f"Количество выбросов: {row['num_outliers']}")
        doc.add_paragraph(f"Процент выбросов: {row['percent_outliers']}%")
        doc.add_paragraph(f"Значения выбросов: {row['outlier_values']}")
        conclusion = ("Выбросов немного." if row['percent_outliers'] < 5
                      else "Присутствует значительное количество выбросов.")
        doc.add_paragraph(f"Вывод: {conclusion}", style='Quote')
        doc.add_paragraph()
def export_sigmas_report(sigmas_res: pd.DataFrame, doc: Document):
    """
    Добавляет в doc главу с анализом выбросов по правилу 3 сигм.
    """
    doc.add_heading("Глава 5. Анализ выбросов по правилу 3 сигм", level=1)
    for row in sigmas_res.to_dict('records'):
        doc.add_heading(f"Столбец: {row['column']}", level=2)
        doc.add_paragraph(f"Среднее: {row['mean']}")
        doc.add_paragraph(f"Стандартное отклонение: {row['std']}")
        doc.add_paragraph(f"Допустимый диапазон: от {row['lower_bound']} до {row['upper_bound']}")
        doc.add_paragraph(f"Количество выбросов: {row['num_outliers']}")
        doc.add_paragraph(f"Процент выбросов: {row['percent_outliers']}%")
        doc.add_paragraph(f"Значения выбросов: {row['outlier_values']}")
        conclusion = ("Выбросов немного." if row['percent_outliers'] < 5
                      else "Обнаружено значительное количество выбросов.")
        doc.add_paragraph(f"Вывод: {conclusion}", style='Quote')
        doc.add_paragraph()

def export_grabbs_report(grabbs_res: pd.DataFrame, doc: Document):
    """
    Добавляет в doc главу с результатами теста Граббса.
    """
    doc.add_heading("Глава 6. Тест Граббса на выбросы", level=1)
    doc.add_paragraph("⚠ Перед применением теста Граббса убедитесь, что данные имеют нормальное распределение.\n")

    for row in grabbs_res.to_dict('records'):
        doc.add_heading(f"Столбец: {row['column']}", level=2)
        doc.add_paragraph(f"Среднее: {row['mean']}")
        doc.add_paragraph(f"Стандартное отклонение: {row['std']}")
        doc.add_paragraph(f"Статистика G: {row['G']}")
        doc.add_paragraph(f"Критическое значение G: {row['G_crit']}")
        if row['outlier_value'] != '':
            doc.add_paragraph(f"Обнаруженный выброс: {row['outlier_value']}")
        doc.add_paragraph(f"Вывод: {row['conclusion']}", style='Quote')
        doc.add_paragraph()
def export_correlation_report(corr_matrix: pd.DataFrame, r2_matrix: pd.DataFrame,
                              t_matrix: pd.DataFrame, significance_matrix: pd.DataFrame,
                              doc: Document, alpha: float = 0.05):
    """
    Добавляет в doc главу с анализом корреляции между числовыми признаками.
    """

    def add_matrix_table(title: str, matrix: pd.DataFrame, formatter=str):
        doc.add_heading(title, level=2)
        table = doc.add_table(rows=1 + len(matrix), cols=1 + len(matrix.columns))
        table.style = 'Table Grid'

        # Заголовки
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = ""
        for j, col in enumerate(matrix.columns):
            hdr_cells[j + 1].text = str(col)

        # Данные
        for i, row_name in enumerate(matrix.index):
            row_cells = table.rows[i + 1].cells
            row_cells[0].text = str(row_name)
            for j, col in enumerate(matrix.columns):
                val = matrix.loc[row_name, col]
                row_cells[j + 1].text = formatter(val)

    doc.add_heading("Глава 6. Анализ корреляции между числовыми признаками", level=1)

    add_matrix_table("Матрица коэффициентов корреляции (r):", corr_matrix,
                     lambda v: f"{v:.3f}")
    add_matrix_table("Матрица детерминации (R²):", r2_matrix,
                     lambda v: f"{v:.3f}")
    add_matrix_table("Матрица t-статистик:", t_matrix,
                     lambda v: f"{v:.3f}")
    add_matrix_table(f"Матрица значимости (t_op > t_critical, α = {alpha}):",
                     significance_matrix, lambda v: "ЗН" if v else "НЗ")




def export_regression_report(models: dict, doc: Document):
    """
    Добавляет в doc главу с результатами построения и анализа регрессионных моделей.
    models: словарь с параметрами моделей, ключ — (X, Y), значения — dict с параметрами ('k', 'b', 'σ_ост', 'η²', 'TSS', 'ESS', 'RSS')
    """
    doc.add_heading("Глава 7. Анализ качества моделей линейной регрессии", level=1)

    for (x_col, y_col), params in models.items():
        doc.add_heading(f"Модель: {y_col} = k * {x_col} + b", level=2)
        doc.add_paragraph(f"Коэффициент наклона (k): {params['k']:.4f}")
        doc.add_paragraph(f"Свободный член (b): {params['b']:.4f}")
        doc.add_paragraph(f"Среднеквадратичное отклонение остатков (σ_ост): {params['σ_ост']:.4f}")
        doc.add_paragraph(f"Коэффициент детерминации (η²): {params['η²']:.4f}")
        doc.add_paragraph(f"Полная сумма квадратов (TSS): {params['TSS']:.4f}")
        doc.add_paragraph(f"Объяснённая сумма квадратов (ESS): {params['ESS']:.4f}")
        doc.add_paragraph(f"Остаточная сумма квадратов (RSS): {params['RSS']:.4f}")

        tss = params['TSS']
        ess = params['ESS']
        rss = params['RSS']
        check = "Да" if np.isclose(tss, ess + rss, atol=1e-4) else "Нет"
        doc.add_paragraph(f"Проверка: TSS ≈ ESS + RSS → {tss:.4f} ≈ {ess + rss:.4f} → Корректность: {check}")
        doc.add_paragraph()


def create(df):

    cr = CheckRandom()
    co = CheckOutliers()
    # Сначала получаем результаты тестов:
    pirs_df = cr.check_pirs(df)
    median_df = cr.check_median(df)
    cr.build_hist(df)
    # Создаём документ и добавляем главы
    doc = Document()
    doc.add_heading("Статистический отчёт", level=0)
    out = co.check_quantiles(df)
    out2 = co.check_sigmas(df)
    out3 = co.check_grabbs(df)

    # Корреляционный анализ
    corr_matrix, r2_matrix, t_matrix, significance_matrix = CheckCorell.check_correlation(df)
    models = CheckCorell.build_regression_models(df, significance_matrix)
    export_pirs_report(pirs_df, doc)
    export_median_report(median_df, doc)
    export_histograms_report(doc)
    export_outliers_report(out, doc)
    export_sigmas_report(out2, doc)
    export_grabbs_report(out3, doc)

    export_correlation_report(corr_matrix, r2_matrix, t_matrix, significance_matrix, doc)
    export_regression_report(models, doc)
    # Сохраняем
    doc.save("stat_report.docx")
    print("Отчёт сохранён в stat_report.docx")