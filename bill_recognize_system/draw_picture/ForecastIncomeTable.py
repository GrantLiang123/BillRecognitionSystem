import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from datetime import datetime, timedelta


class ForecastIncomeTable:
    def __init__(self):
        # 使用支持中文的字体
        plt.rcParams['font.family'] = 'Arial Unicode MS'

    def create_line_chart(self, data1, data2):
        # 按照日期对data1进行排序
        data1.sort(key=lambda x: datetime.strptime(x[0], '%Y年%m月%d日'))

        # 合并data1中日期相同的数据,并将收入转换为float类型
        merged_data1 = {}
        for date, income in data1:
            if date in merged_data1:
                merged_data1[date] += float(income)
            else:
                merged_data1[date] = float(income)
        data1 = list(merged_data1.items())

        # 给data2补上日期
        last_date = datetime.strptime(data1[-1][0], '%Y年%m月%d日')
        data2_with_date = []
        for income in data2:
            last_date += timedelta(days=1)
            data2_with_date.append([last_date.strftime('%Y年%m月%d日'), float(income)])
        data2 = data2_with_date

        # 从输入数据中提取时间和收入
        times1 = [row[0] for row in data1]
        incomes1 = [row[1] for row in data1]
        times2 = [row[0] for row in data2]
        incomes2 = [row[1] for row in data2]

        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制data1的折线图
        line1, = ax.plot(times1, incomes1, marker='o', linewidth=2, markersize=8, color='#2c7fb8')

        # 绘制data2的折线图,起点设置为data1的最后一个数据点
        line2, = ax.plot(times2, incomes2, marker='o', linewidth=2, markersize=8, color='#ff0000')
        line2.set_data([times1[-1]] + times2, [incomes1[-1]] + incomes2)

        # 在每个数据点上添加数值标签
        for i, income in enumerate(incomes1):
            ax.annotate(f'{income}', (times1[i], income), textcoords="offset points", xytext=(0, 10), ha='center')
        for i, income in enumerate(incomes2):
            ax.annotate(f'{income}', (times2[i], income), textcoords="offset points", xytext=(0, 10), ha='center')

        # 设置图像标题和轴标签
        ax.set_title('收入预测曲线', fontsize=18, fontweight='bold', color='#2c7fb8')
        ax.set_xlabel('日期', fontsize=14)
        ax.set_ylabel('收入', fontsize=14)

        # 设置图例
        ax.legend([line1, line2], ['过往数据', '预测'], loc='upper left', fontsize=12)

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
        ['2023年01月9日', "1000"],
        ['2023年2月5日', "1500"],
        ['2023年01月9日', "1200"],
        ['2023年04月3日', "1800"],
        ['2023年03月4日', "2000"],
    ]
    data2 = ['2200', '2500', '2800', '3000']

    forecast_expenditure = ForecastExpenditure()
    img_buffer = forecast_expenditure.create_line_chart(data1, data2)

    # 将图像保存到文件
    with open('forcast_expenditure_chart.png', 'wb') as f:
        f.write(img_buffer.getvalue())