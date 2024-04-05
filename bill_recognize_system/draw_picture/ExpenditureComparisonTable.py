import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
# matplotlib.font_manager._rebuild()
from collections import defaultdict

class ExpenditureComparisonTable:
    def __init__(self):
        # 使用支持中文的字体
        plt.rcParams['font.family'] = 'Arial Unicode MS'

    def create_comparison_chart(self, data1, data2):
        # 检查输入数据的维度是否正确
        if len(data1) == 0 or len(data1[0]) != 2 or len(data2) == 0 or len(data2[0]) != 2:
            raise ValueError("输入的列表维度应为 n×2")

        # 将数据转换为字典,以日期为键,收入为值
        income_dict1 = defaultdict(float)
        for date, income in data1:
            day = date.split('-')[-1]  # 提取日期中的日子部分
            income_dict1[day] = income

        income_dict2 = defaultdict(float)
        for date, income in data2:
            day = date.split('-')[-1]  # 提取日期中的日子部分
            income_dict2[day] = income

        # 获取所有唯一的日子
        days = sorted(set(income_dict1.keys()) | set(income_dict2.keys()))

        # 创建收入列表,按日子顺序排列
        incomes1 = [income_dict1[day] for day in days]
        incomes2 = [income_dict2[day] for day in days]

        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制折线图
        line1, = ax.plot(days, incomes1, marker='o', linewidth=2, markersize=8, color='#2c7fb8', label='数据1')
        line2, = ax.plot(days, incomes2, marker='o', linewidth=2, markersize=8, color='#ff7f0e', label='数据2')

        # 在每个数据点上添加数值标签
        for i, income in enumerate(incomes1):
            ax.annotate(f'{income}', (days[i], income), textcoords="offset points", xytext=(0, 10), ha='center')
        for i, income in enumerate(incomes2):
            ax.annotate(f'{income}', (days[i], income), textcoords="offset points", xytext=(0, -15), ha='center')

        # 设置图像标题和轴标签
        ax.set_title('支出对比', fontsize=18, fontweight='bold', color='#2c7fb8')
        ax.set_xlabel('日期', fontsize=14)
        ax.set_ylabel('支出', fontsize=14)

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
        ['2023-01-01', 1000],
        ['2023-01-02', 1500],
        ['2023-01-03', 1200],
        ['2023-01-04', 1800],
        ['2023-01-05', 2000],
    ]

    data2 = [
        ['2023-02-01', 800],
        ['2023-02-02', 1200],
        ['2023-02-03', 1000],
        ['2023-02-04', 1500],
        ['2023-02-06', 1800],
    ]
    a=ExpenditureComparisonTable()
    img_buffer = a.create_comparison_chart(data1, data2)

    # 将图像保存到文件
    with open('expenditure_comparison.png', 'wb') as f:
        f.write(img_buffer.getvalue())