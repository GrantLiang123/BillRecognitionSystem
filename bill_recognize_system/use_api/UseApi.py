import sys

sys.path.append('/home/lighthouse/BillRecognitionSystem')
from flask import Flask, jsonify, request, send_file
#from flask_cors import CORS
import bill_recognize_system.draw_picture.LedgerTable as Le
import bill_recognize_system.draw_picture.IncomeTable as In
import bill_recognize_system.draw_picture.IncomeComparisonTable as IC
import bill_recognize_system.draw_picture.IncomeAndExpenditureStatement as IE
import bill_recognize_system.draw_picture.ForecastIncomeTable as FI
import bill_recognize_system.draw_picture.ForecastExpenditureTable as FE
import bill_recognize_system.draw_picture.ExpenditureTable as Ex
import bill_recognize_system.draw_picture.ExpenditureComparisonTable as EC
import bill_recognize_system.data_operate.generate_xlsx as GX
from bill_recognize_system.data_calculate.regression_prediction import bill_predict
import json

# from bill_recognize_system.data_calculate.regression_prediction import bill_predict


app = Flask(__name__)
#
# CORS(app)


@app.route('/MaNongBbq/ledgerTable', methods=['POST'])
def create_ledger_table():
    data_1 = request.json['data']

    creator = Le.LedgerTable()
    image_data = creator.create_table_image(data_1)

    return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/incomeTable', methods=['POST'])
def create_income_table():
    data = request.json['data']

    creator = In.IncomeTable()
    image_data = creator.create_line_chart(data)

    return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/incomeComparison', methods=['POST'])
def create_income_comparison():
    data1 = request.json['data']

    # 创建 IncomeComparisonTable 实例
    comparison_table = IC.IncomeComparisonTable()

    # 调用 create_comparison_chart 方法生成图像
    img_buffer = comparison_table.create_comparison_chart(data1)

    # 将图像作为文件发送给客户端
    return send_file(img_buffer, mimetype='image/png')


@app.route('/MaNongBbq/incomeExpenditure', methods=['POST'])
def create_income_expenditure():
    data = request.json['data']

    creator = IE.IncomeAndExpenditureStatement()
    image_data = creator.create_line_chart(data)

    return send_file(image_data, mimetype='image/png')


# @app.route('/MaNongBbq/forecastIncome', methods=['POST'])
# def create_forecast_income():
#     data = request.json['data']
#
#
#     creator = FI.ForecastIncomeTable()
#     image_data = creator.create_line_chart(data)
#
#     return send_file(image_data, mimetype='image/png')


# @app.route('/MaNongBbq/forecastExpenditure', methods=['POST'])
# def create_forecast_expenditure():
#     data = request.json['data']
#
#     creator = FE.ForecastExpenditure()
#     image_data = creator.create_line_chart(data)
#
#     return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/expenditureTable', methods=['POST'])
def create_expenditure_table():
    data = request.json['data']

    creator = Ex.ExpenditureTable()
    image_data = creator.create_line_chart(data)

    return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/expenditureComparison', methods=['POST'])
def create_expenditure_comparison():
    data1 = request.json['data']

    # 创建 IncomeComparisonTable 实例
    comparison_table = EC.ExpenditureComparisonTable()

    # 调用 create_comparison_chart 方法生成图像
    img_buffer = comparison_table.create_comparison_chart(data1)

    # 将图像作为文件发送给客户端
    return send_file(img_buffer, mimetype='image/png')


@app.route('/MaNongBbq/regressionPredict', methods=['POST'])
def regression_predict():
    # 从请求的 JSON 数据中获取参数
    y_real = request.json['y_real']
    is_complex_model = request.json['is_complex_model']
    forecast_days = request.json['forecast_days']
    use_model_name = request.json['use_model_name']
    is_income = request.json['is_income']
    date_time = request.json['date_time']

    if is_complex_model is None:
        is_complex_model = False
    if forecast_days is None:
        forecast_days = 3
    if use_model_name is None:
        use_model_name = 'linear_regression'

    mer_list = list(zip(date_time, y_real))
    float_list = [float(item) for item in y_real]

    result = bill_predict(float_list, is_complex_model, forecast_days, use_model_name)
    # 将结果作为 JSON 响应返回
    result_dict = json.loads(result)
    y_predict = result_dict['y_predict']
    # print(y_predict)

    if is_income is True:
        a1 = FI.ForecastIncomeTable()
        re1 = a1.create_line_chart(mer_list, y_predict)
        return send_file(re1, mimetype='image/png')
    else:
        a2 = FE.ForecastExpenditure()
        re2 = a2.create_line_chart(mer_list, y_predict)
        return send_file(re2, mimetype='image/png')


@app.route('/MaNongBbq/generateXlsx01', methods=['POST'])
def generate_xlsx01():
    if request.json and isinstance(request.json, dict) and 'data' in request.json:
        data = request.json['data']
        a = GX.GenerateExcel()
        file_stream = a.generate_excel_1(data, 'example.xlsx')
        return send_file(file_stream, attachment_filename='example.xlsx', as_attachment=True)
    else:
        return 'Invalid request data', 400


@app.route('/MaNongBbq/generateXlsx02', methods=['POST'])
def generate_xlsx02():
    if request.json and isinstance(request.json, dict) and 'data' in request.json:
        data = request.json['data']
        a = GX.GenerateExcel()
        file_stream = a.generate_excel_2(data, 'example.xlsx')
        return send_file(file_stream, attachment_filename='example.xlsx', as_attachment=True)
    else:
        return 'Invalid request data', 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8809, debug=True)
