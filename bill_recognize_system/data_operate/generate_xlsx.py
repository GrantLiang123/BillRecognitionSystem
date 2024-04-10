from openpyxl import Workbook
from io import BytesIO
import matplotlib.pyplot as plt

class GenerateExcel:
    def __init__(self):
        # 使用支持中文的字体
        plt.rcParams['font.family'] = 'Arial Unicode MS'

    def generate_excel_1(self, data, filename='output.xlsx'):
        # 创建一个新的工作簿
        workbook = Workbook()
        sheet = workbook.active
        # 定义表头
        headers = ['开票日期', '价税合计小写', '票据类型（中文）', '货物或服务名称', '发票号码', '发票代码',
                   '纳税人识别号', '增值税发票No号码', '税额合计', '开票人', '是否为收入']
        sheet.append(headers)

        if data and len(data) > 0 and isinstance(data[0], dict):
            # 写入表头
            headers = list(data[0].keys())
            sheet.append(headers)

            # 写入数据
            for row in data:
                sheet.append(list(row.values()))
        elif data and isinstance(data[0], list):
            # 如果data[0]是列表,则直接将数据写入工作表
            for row in data:
                sheet.append(row)
        else:
            raise ValueError("数据格式不正确或数据为空")

        # 将Excel文件写入内存中的字节流
        file_stream = BytesIO()
        workbook.save(file_stream)
        file_stream.seek(0)

        # 返回文件的字节流
        return file_stream

    def generate_excel_2(self, data, filename='output.xlsx'):
        # 创建一个新的工作簿
        workbook = Workbook()
        sheet = workbook.active
        # 定义表头
        headers = ['发票时间', '价税合计', '票据类型（中文）',  '发票号码', '发票代码',
                   '纳税人识别号', '是否为收入']
        sheet.append(headers)

        if data and len(data) > 0 and isinstance(data[0], dict):
            # 写入表头
            headers = list(data[0].keys())
            sheet.append(headers)

            # 写入数据
            for row in data:
                sheet.append(list(row.values()))
        elif data and isinstance(data[0], list):
            # 如果data[0]是列表,则直接将数据写入工作表
            for row in data:
                sheet.append(row)
        else:
            raise ValueError("数据格式不正确或数据为空")

        # 将Excel文件写入内存中的字节流
        file_stream = BytesIO()
        workbook.save(file_stream)
        file_stream.seek(0)

        # 返回文件的字节流
        return file_stream