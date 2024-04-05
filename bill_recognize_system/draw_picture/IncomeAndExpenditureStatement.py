import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
matplotlib.font_manager._rebuild()

class IncomeAndExpenditureStatement:
    def __init__(self):
        # 使用支持中文的字体
        plt.rcParams['font.family'] = 'Arial Unicode MS'
    def create_line_chart(self,data):
        # 检查输入数据的维度是否正确
        if len(data) == 0 or len(data[0]) != 2:
            raise ValueError("输入的列表维度应为 n×2")

        # 从输入数据中提取时间和收入
        times = [row[0] for row in data]
        incomes = [row[1] for row in data]

        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制折线图
        ax.plot(times, incomes, marker='o')

        # 设置图像标题和轴标签
        ax.set_title('收支曲线')
        ax.set_xlabel('时间')
        ax.set_ylabel('收入')

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
#     with open('income_expenditure_chart.png', 'wb') as f:
#         f.write(img_buffer.getvalue())