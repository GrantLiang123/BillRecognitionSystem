from openpyxl import Workbook
from io import BytesIO
import matplotlib.pyplot as plt


class GenerateExcel:
    def __init__(self):
        # 使用支持中文的字体
        plt.rcParams['font.family'] = 'Arial Unicode MS'

    def generate_excel(self,data, filename='output.xlsx'):
        # 创建一个新的工作簿
        workbook = Workbook()
        sheet = workbook.active

        # 写入表头
        headers = list(data[0].keys())
        sheet.append(headers)

        # 写入数据
        for row in data:
            sheet.append(list(row.values()))

        # 将Excel文件写入内存中的字节流
        file_stream = BytesIO()
        workbook.save(file_stream)
        file_stream.seek(0)

        # 返回文件的字节流
        return file_stream