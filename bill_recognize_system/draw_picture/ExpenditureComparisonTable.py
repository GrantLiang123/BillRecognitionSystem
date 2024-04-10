import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from collections import defaultdict
from datetime import datetime


class ExpenditureComparisonTable:
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
        data_current_month = [row for row in data if
                              datetime.strptime(row[0], '%Y年%m月%d日').year == current_year and datetime.strptime(
                                  row[0], '%Y年%m月%d日').month == current_month]
        data_previous_month = [row for row in data if
                               datetime.strptime(row[0], '%Y年%m月%d日').year == current_year and datetime.strptime(
                                   row[0], '%Y年%m月%d日').month == current_month - 1]

        # 过滤掉日期或收入为空的数据
        data_current_month = [row for row in data_current_month if row[0] and row[1]]
        data_previous_month = [row for row in data_previous_month if row[0] and row[1]]

        # 将收入转换为浮点数
        data_current_month = [[row[0], float(row[1])] for row in data_current_month]
        data_previous_month = [[row[0], float(row[1])] for row in data_previous_month]

        # 合并相同日期的收入
        merged_data_current_month = defaultdict(float)
        for row in data_current_month:
            merged_data_current_month[datetime.strptime(row[0], '%Y年%m月%d日').day] += row[1]

        merged_data_previous_month = defaultdict(float)
        for row in data_previous_month:
            merged_data_previous_month[datetime.strptime(row[0], '%Y年%m月%d日').day] += row[1]

        # 创建一个包含30天的列表
        days = list(range(1, 32))

        # 创建收入列表,按日期顺序排列
        incomes_current_month = [merged_data_current_month[day] for day in days]
        incomes_previous_month = [merged_data_previous_month[day] for day in days]

        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(12, 6))

        # 设置柱状图的宽度
        bar_width = 0.35

        # 绘制柱状图
        bars_current_month = ax.bar([day - bar_width / 2 for day in days], incomes_current_month, width=bar_width,
                                    color='#2c7fb8', label=f'{current_year}年{current_month}月')
        bars_previous_month = ax.bar([day + bar_width / 2 for day in days], incomes_previous_month, width=bar_width,
                                     color='#ff7f0e', label=f'{current_year}年{current_month - 1}月')

        # 在柱子上添加数据标签
        for bar in bars_current_month:
            height = bar.get_height()
            if height != 0:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        for bar in bars_previous_month:
            height = bar.get_height()
            if height != 0:
                ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        # 设置图像标题和轴标签
        ax.set_title('支出对比', fontsize=18, fontweight='bold', color='#2c7fb8')
        ax.set_xlabel('日期', fontsize=14)
        ax.set_ylabel('支出', fontsize=14)

        # 设置x轴刻度
        ax.set_xticks(days)
        ax.set_xticklabels(days)

        # 设置图例
        ax.legend(fontsize=12)

        # 设置网格线
        ax.grid(True, linestyle='--', alpha=0.7)

        # 设置背景颜色
        ax.set_facecolor('#f0f0f0')

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
        ['2024年04月3日', "1800"],
        ['2024年04月4日', "2000"],
        ['2024年04月9日', "800"],
        ['', "2000"]
    ]
    a = ExpenditureComparisonTable()
    img_buffer = a.create_comparison_chart(data1)

    # 将图像保存到文件
    with open('expenditure_comparison.png', 'wb') as f:
        f.write(img_buffer.getvalue())
