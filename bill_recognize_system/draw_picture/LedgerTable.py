import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from datetime import datetime

class LedgerTable:
    def __init__(self):
        plt.rcParams['font.family'] = 'Arial Unicode MS'

    def create_table_image(self, data1, data2):
        # 提取年份和月份
        year, month = data1[0], data1[1]

        # 筛选与data1相同年月的数据,并按日期排序
        filtered_data = [row for row in data2 if datetime.strptime(row[0], '%Y年%m月%d日').year == year and datetime.strptime(row[0], '%Y年%m月%d日').month == month]
        sorted_data = sorted(filtered_data, key=lambda x: datetime.strptime(x[0], '%Y年%m月%d日'), reverse=True)

        # 添加表头
        header = ['时间', '摘要', '会计科目', '收入', '支出', '总计']
        table_data = [header] + sorted_data

        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.8))

        # 创建表格
        table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1, 2)

        # 根据单元格内容的长度自动调整列宽
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table_data)]
        for i in range(len(col_widths)):
            table.auto_set_column_width(i)

        # 隐藏坐标轴
        ax.axis('off')

        # 将图像转换为 PNG 格式的字节流
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.05)
        img_buffer.seek(0)

        # 关闭图像以释放内存
        plt.close(fig)

        return img_buffer

if __name__ == "__main__":
    data1 = [2023, 5]
    data2 = [
        ['2023-05-20', '购买设备', '固定资产', '', '10000', '10000'],
        ['2023-05-21', '销售商品', '主营业务收入', '5000', '', '15000'],
        ['2023-05-22', '支付工资', '应付职工薪酬', '', '8000', '7000'],
        ['2023-04-25', '购买原材料', '原材料', '', '5000', '2000'],
        ['2023-06-01', '销售商品', '主营业务收入', '8000', '', '10000'],
    ]
    a = LedgerTable()
    img_buffer = a.create_table_image(data1, data2)

    # 将图像保存到文件
    with open('table.png', 'wb') as f:
        f.write(img_buffer.getvalue())