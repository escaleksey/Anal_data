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


def create(df):

    cr = CheckRandom()
    # Сначала получаем результаты тестов:
    pirs_df = cr.check_pirs(df)
    median_df = cr.check_median(df)
    cr.build_hist(df)
    # Создаём документ и добавляем главы
    doc = Document()
    doc.add_heading("Статистический отчёт", level=0)

    export_pirs_report(pirs_df, doc)
    export_median_report(median_df, doc)
    export_histograms_report(doc)

    # Сохраняем
    doc.save("stat_report.docx")
    print("Отчёт сохранён в stat_report.docx")