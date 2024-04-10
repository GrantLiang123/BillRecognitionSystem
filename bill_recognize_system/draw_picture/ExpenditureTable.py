import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from datetime import datetime

class ExpenditureTable:
    def __init__(self):
        # 使用支持中文的字体
        plt.rcParams['font.family'] = 'Arial Unicode MS'

    def create_line_chart(self, data):
        # 过滤掉日期或收入为空的数据
        data = [row for row in data if row[0] and row[1]]
        # 将收入转换为浮点数
        data = [[row[0], float(row[1])] for row in data]

        # 将日期字符串转换为日期对象,并按日期排序
        data = sorted(data, key=lambda x: datetime.strptime(x[0], '%Y年%m月%d日'))

        # 合并相同日期的收入
        merged_data = {}
        for row in data:
            if row[0] in merged_data:
                merged_data[row[0]] += row[1]
            else:
                merged_data[row[0]] = row[1]

        # 将合并后的数据转换为列表
        data = [[date, income] for date, income in merged_data.items()]

        # 从输入数据中提取时间和收入
        times = [row[0] for row in data]
        incomes = [row[1] for row in data]

        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制折线图
        line, = ax.plot(times, incomes, marker='o', linewidth=2, markersize=8, color='#2c7fb8')

        # 在每个数据点上添加数值标签
        for i, income in enumerate(incomes):
            ax.annotate(f'{income}', (times[i], income), textcoords="offset points", xytext=(0, 10), ha='center')

        # 设置图像标题和轴标签
        ax.set_title('支出曲线', fontsize=18, fontweight='bold', color='#2c7fb8')
        ax.set_xlabel('日期', fontsize=14)
        ax.set_ylabel('支出', fontsize=14)

        # 设置图例
        ax.legend([line], ['支出'], loc='upper left', fontsize=12)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)

        # 设置背景颜色
        ax.set_facecolor('#f0f0f0')

        # 自动调整刻度标签的旋转角度
        plt.xticks(rotation=45, ha='right')

        # 自动调整图像布局
        plt.tight_layout()

        # 将图像转换为 PNG 格式的字节流
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # 关闭图像以释放内存
        plt.close(fig)

        return img_buffer

# if __name__ == "__main__":
#     data = [
#         ['2023-01', 1000],
#         ['2023-02', 1500],
#         ['2023-03', 1200],
#         ['2023-04', 1800],
#         ['2023-05', 2000],
#     ]
#
#     img_buffer = create_line_chart(data)
#
#     # 将图像保存到文件
#     with open('expenditure_chart.png', 'wb') as f:
#         f.write(img_buffer.getvalue())