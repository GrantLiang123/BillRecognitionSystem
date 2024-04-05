import io
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
matplotlib.font_manager._rebuild()

class LedgerTable:
    def __init__(self):
        # font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        # for font_path in font_list:
        #     print(font_path)
        # 使用支持中文的字体
        #font_path = '/usr/share/fonts/ArialUnicodeMS.ttf'
        #self.font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = 'Arial Unicode MS'
    def create_table_image(self, data):
        # 检查输入数据的维度是否正确
        if len(data) == 0 or len(data[0]) != 7:
            raise ValueError("输入的列表维度应为 n×7")

        # 添加表头
        header = ['序号', '时间', '摘要', '会计科目', '收入', '支出', '总计']
        data.insert(0, header)

        # 创建一个新的图像
        fig, ax = plt.subplots(figsize=(10, len(data) * 0.5))

        # 创建表格
        table = ax.table(cellText=data, cellLoc='center', loc='center')

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)

        # 根据单元格内容的长度自动调整列宽
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*data)]
        for i in range(len(col_widths)):
            table.auto_set_column_width(col_widths[i] / 2)

        # 隐藏坐标轴
        ax.axis('off')

        # 调整表格边距
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

        # 将图像转换为 PNG 格式的字节流
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # 关闭图像以释放内存
        plt.close(fig)

        return img_buffer

# if __name__ == "__main__":
#     data = [
#         ['1', '2023-05-20', '购买设备', '固定资产', '', '10000', '10000'],
#         ['2', '2023-05-21', '销售商品', '主营业务收入', '5000', '', '15000'],
#         ['3', '2023-05-22', '支付工资', '应付职工薪酬', '', '8000', '7000'],
#     ]
#
#     img_buffer = create_table_image(data)
#
#     # 将图像保存到文件
#     with open('table.png', 'wb') as f:
#         f.write(img_buffer.getvalue())