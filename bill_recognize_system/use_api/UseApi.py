import sys

sys.path.append('/home/lighthouse/BillRecognitionSystem')
from flask import Flask, jsonify, request, send_file
# from flask_cors import CORS
import bill_recognize_system.draw_picture.LedgerTable as Le;
import bill_recognize_system.draw_picture.IncomeTable as In;
import bill_recognize_system.draw_picture.IncomeComparisonTable as IC;
import bill_recognize_system.draw_picture.IncomeAndExpenditureStatement as IE;
import bill_recognize_system.draw_picture.ForecastIncomeTable as FI;
import bill_recognize_system.draw_picture.ForecastExpenditureTable as FE;
import bill_recognize_system.draw_picture.ExpenditureTable as Ex;
import bill_recognize_system.draw_picture.ExpenditureComparisonTable as EC;
from bill_recognize_system.data_calculate.regression_prediction import bill_predict
# from bill_recognize_system.data_calculate.regression_prediction import bill_predict


app = Flask(__name__)


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
    data1 = request.json['data1']
    data2 = request.json['data2']

    # 创建 IncomeComparisonTable 实例
    comparison_table = IC.IncomeComparisonTable()

    # 调用 create_comparison_chart 方法生成图像
    img_buffer = comparison_table.create_comparison_chart(data1, data2)

    # 将图像作为文件发送给客户端
    return send_file(img_buffer, mimetype='image/png')


@app.route('/MaNongBbq/incomeExpenditure', methods=['POST'])
def create_income_expenditure():
    data = request.json['data']

    creator = IE.IncomeAndExpenditureStatement()
    image_data = creator.create_line_chart(data)

    return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/forecastIncome', methods=['POST'])
def create_forecast_income():
    data = request.json['data']

    creator = FI.ForecastIncomeTable()
    image_data = creator.create_line_chart(data)

    return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/forecastExpenditure', methods=['POST'])
def create_forecast_expenditure():
    data = request.json['data']

    creator = FE.ForecastExpenditure()
    image_data = creator.create_line_chart(data)

    return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/expenditureTable', methods=['POST'])
def create_expenditure_table():
    data = request.json['data']

    creator = Ex.ExpenditureTable()
    image_data = creator.create_line_chart(data)

    return send_file(image_data, mimetype='image/png')


@app.route('/MaNongBbq/expenditureComparison', methods=['POST'])
def create_expenditure_comparison():
    data1 = request.json['data1']
    data2 = request.json['data2']

    # 创建 IncomeComparisonTable 实例
    comparison_table = EC.ExpenditureComparisonTable();

    # 调用 create_comparison_chart 方法生成图像
    img_buffer = comparison_table.create_comparison_chart(data1, data2)

    # 将图像作为文件发送给客户端
    return send_file(img_buffer, mimetype='image/png')


@app.route('/MaNongBbq/regressionPredict', methods=['POST'])
def regression_predict():
    # 从请求的 JSON 数据中获取参数
    y_real = request.json['y_real']
    is_complex_model = request.json['is_complex_model']
    forecast_days = request.json['forecast_days']
    use_model_name = request.json['use_model_name']

    # 创建 RegressionPrediction 实例
    result=bill_predict(y_real, is_complex_model, forecast_days,use_model_name)

    # 将结果作为 JSON 响应返回
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8809, debug=True)
