import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from collections import defaultdict
from datetime import datetime

class IncomeComparisonTable:
    def __init__(self):
        # 使用支持中文的字体
        plt.rcParams['font.family'] = 'Arial Unicode MS'

    def create_comparison_chart(self, data):
        # 检查输入数据的维度是否正确
        if len(data) == 0 or len(data[0]) != 2:
            raise ValueError("输入的列表维度应为 n×2")

        # 获取当前时间
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month

        # 过滤掉日期为空的数据
        data = [row for row in data if row[0]]

        # 筛选出本月和上个月的数据
        data_current_month = [row for row in data if datetime.strptime(row[0], '%Y年%m月%d日').year == current_year and datetime.strptime(row[0], '%Y年%m月%d日').month == current_month]
        data_previous_month = [row for row in data if datetime.strptime(row[0], '%Y年%m月%d日').year == current_year and datetime.strptime(row[0], '%Y年%m月%d日').month == current_month - 1]

        # 过滤掉日期或收入为空的数据
        data_current_month = [row for row in data_current_month if row[0] and row[1]]
        data_previous_month = [row for row in data_previous_month if row[0] and row[1]]

        # 将收入转换为浮点数
        data_current_month = [[row[0], float(row[1])] for row in data_current_month]
        data_previous_month = [[row[0], float(row[1])] for row in data_previous_month]

        # 将日期字符串转换为日期对象,并按日期排序
        data_current_month = sorted(data_current_month, key=lambda x: datetime.strptime(x[0], '%Y年%m月%d日'))
        data_previous_month = sorted(data_previous_month, key=lambda x: datetime.strptime(x[0], '%Y年%m月%d日'))

        # 合并相同日期的收入
        merged_data_current_month = {}
        for row in data_current_month:
            if row[0] in merged_data_current_month:
                merged_data_current_month[row[0]] += row[1]
            else:
                merged_data_current_month[row[0]] = row[1]

        merged_data_previous_month = {}
        for row in data_previous_month:
            if row[0] in merged_data_previous_month:
                merged_data_previous_month[row[0]] += row[1]
            else:
                merged_data_previous_month[row[0]] = row[1]

        # 获取所有日期
        all_dates = sorted(set(merged_data_current_month.keys()) | set(merged_data_previous_month.keys()))

        # 创建收入列表,按日子顺序排列
        incomes_current_month = [merged_data_current_month.get(date, 0) for date in all_dates]
        incomes_previous_month = [merged_data_previous_month.get(date, 0) for date in all_dates]

        # 提取日期的日部分
        days = [datetime.strptime(date, '%Y年%m月%d日').strftime('%d') for date in all_dates]


        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制折线图
        line1, = ax.plot(days, incomes_current_month, marker='o', linewidth=2, markersize=8, color='#2c7fb8',
                         label='本月')
        line2, = ax.plot(days, incomes_previous_month, marker='o', linewidth=2, markersize=8, color='#ff7f0e',
                         label='上月')

        # 在每个数据点上添加数值标签
        for i, income in enumerate(incomes_current_month):
            ax.annotate(f'{income}', (days[i], income), textcoords="offset points", xytext=(0, 10), ha='center')
        for i, income in enumerate(incomes_previous_month):
            ax.annotate(f'{income}', (days[i], income), textcoords="offset points", xytext=(0, -15), ha='center')


        # 设置图像标题和轴标签
        ax.set_title('收入对比', fontsize=18, fontweight='bold', color='#2c7fb8')
        ax.set_xlabel('日期', fontsize=14)
        ax.set_ylabel('收入', fontsize=14)

        # 设置图例
        ax.legend(handles=[line1, line2], loc='upper left', fontsize=12)

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

if __name__ == "__main__":
    data1 = [
        ['2024年01月9日', "1000"],
        ['2024年3月4日', "1500"],
        ['2024年03月9日', "1200"],
        ['2024年04月3日', "1800"],
        ['2024年04月4日', "2000"],
        ['2024年04月9日', "800"],
        ['',"2000"]
    ]

    a=IncomeComparisonTable()
    img_buffer = a.create_comparison_chart(data1)

    # 将图像保存到文件
    with open('income_comparison.png', 'wb') as f:
        f.write(img_buffer.getvalue())