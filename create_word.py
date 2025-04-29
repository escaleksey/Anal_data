from utils import CheckOutliers
from utils.check_random import CheckRandom
from docx import Document
from docx.shared import Inches
import pandas as pd
import numpy as np
from scipy.stats import chi2

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
    export_pirs_report(pirs_df, doc)
    export_median_report(median_df, doc)
    export_histograms_report(doc)
    export_outliers_report(out, doc)
    export_sigmas_report(out2, doc)
    export_grabbs_report(out3, doc)
    # Сохраняем
    doc.save("stat_report.docx")
    print("Отчёт сохранён в stat_report.docx")