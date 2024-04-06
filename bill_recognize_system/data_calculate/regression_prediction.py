"""
Include some class and methods about regression algorithms.
"""

# coding : utf-8
# Time : 2024/3/26
# Author : Liang Zi Fan
# File : regression_prediction.py
# Software : Pycharm

import json
from enum import Enum
import numpy as np  # numpy库
from sklearn.ensemble import GradientBoostingRegressor  # 集成算法
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR  # SVM中的回归算法


def data_ascension(x_initial: list, degree: int = 2):
    """
    对传入的X值进行升维。
    :param x_initial: X值的列表
    :param degree: 目标维度，默认为2，需要大于等于2，维度越高，精度越高，但开销会指数级增长。
    :return: 升维后的多为列表
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=False)
    x_middle = np.array(x_initial).reshape(-1, 1)
    x_train = poly.fit_transform(x_middle)
    return x_train


def value_accuracy(y_test_pre: list, y_test: list):
    """
    比较标准值和预测值返回预测精度。

    精度计算方法：标准值与预测值之差的绝对值除以标准值的连乘，数值大于0，
    数值越小精度越高。
    :param y_test_pre:Y值的预测值
    :param y_test:Y值的标准值
    :return:预测精度
    """
    accuracy = 1.0
    for i in range(len(y_test)):
        accuracy *= (abs(y_test[i] - y_test_pre[i])) / y_test[i]
    return accuracy


class ModelName(Enum):
    """
    模型名称的枚举类。

    枚举值：
    --------
    - bayesian_ridge 贝叶斯岭回归模型
    - linear_regression 线性回归模型
    - elastic_net 弹性网络回归模型
    - svr 向量机回归模型
    - gradient_boosting_regressor 梯度增强回归模型
    """
    bayesian_ridge = 1
    linear_regression = 2
    elastic_net = 3
    svr = 4
    gradient_boosting_regressor = 5


class RegressionPrediction:
    """
    This is a class about regression algorithms.

    这是回归预测的类，其中包含两个方法：

    - def train_model_easy(self, y_train_initial: list) -> str  简单回归预测
    - def train_model_complex(self, y_train_initial: list) -> str  复杂回归预测
    二者都可以进行分析预测，在数据简单的情况下，简单回归与复杂回归都具有较好
    的预测精度，复杂回归开销较大。在数据复杂的情况下，简单回归的预测精度随杂乱
    程度的递增而递减（非线性），且开销指数级上升。

    你可以在创建类的时候传递一下参数：

    - forecast_days 预测天数，默认为3天
    - use_model_name 使用模型名称，默认为线性回归模型
    """

    def __init__(
            self,
            *,
            forecast_days: int = 3,
            use_model_name='linear_regression',
    ):
        """
        初始化函数。
        :param forecast_days: 预测天数，默认为3，需要大于0。
        :param use_model_name: 使用的模型名称，传入字符串，自动转换为枚举类型，默认为线性回归模型。
        """
        self.model_br = BayesianRidge()  # 贝叶斯岭回归模型对象
        self.model_lr = LinearRegression()  # 普通线性回归模型对象
        self.model_etc = ElasticNet()  # 弹性网络回归模型对象
        self.model_svr = SVR()  # 支持向量机回归模型对象
        self.model_gbr = GradientBoostingRegressor()  # 梯度增强回归模型对象
        self.forecast_days: int = forecast_days  # 预测天数
        self.use_model_name = ModelName[use_model_name]  # 使用的回归模型
        self.return_list = ['y_predict', 'is_correct', 'predict_accuracy', 'model_name']  # API返回的JSON标签
    #
    # def select_model(self, use_model_name):
    #     """
    #     通过枚举值，返回对应的模型对象。
    #     :param use_model_name: 使用的模型的枚举值。
    #     :return: 模型对象
    #     :rtype BayesianRidge | LinearRegression | ElasticNet | SVR | GradientBoostingRegressor
    #     """
    #     match use_model_name:
    #         case ModelName.bayesian_ridge:
    #             return self.model_br
    #         case ModelName.linear_regression:
    #             return self.model_lr
    #         case ModelName.elastic_net:
    #             return self.model_etc
    #         case ModelName.svr:
    #             return self.model_svr
    #         case ModelName.gradient_boosting_regressor:
    #             return self.model_gbr

    def select_model(self, use_model_name):
        """
        通过枚举值,返回对应的模型对象。
        :param use_model_name: 使用的模型的枚举值。
        :return: 模型对象
        :rtype BayesianRidge | LinearRegression | ElasticNet | SVR | GradientBoostingRegressor
        """
        if use_model_name == ModelName.bayesian_ridge:
            return self.model_br
        elif use_model_name == ModelName.linear_regression:
            return self.model_lr
        elif use_model_name == ModelName.elastic_net:
            return self.model_etc
        elif use_model_name == ModelName.svr:
            return self.model_svr
        elif use_model_name == ModelName.gradient_boosting_regressor:
            return self.model_gbr


    def train_model_easy(self, y_train_initial: list):
        """
        简单回归分析。

        精度较低，与预测维度相关，默认返回2到100维内最小的精度大于90%的结果，
        若没有符合要求的结果，则返回精度最大的结果。

        返回值说明
        ----
        - y_predict：Y值的预测列表
        - is_correct：布尔值，说明结果是否精确，若为True，则精度大于90%；若为False，则没有符合预期的结果，返回精度最大的结果，此时该结果参考价值较小。
        - predict_accuracy 预测的准确率
        - model_name 使用的模型名称
        :param y_train_initial: Y值列表
        :return: JSON列表
        """
        model = self.select_model(self.use_model_name)
        y_train = y_train_initial[:-2]
        y_test = y_train_initial[-2:]
        x_initial = list(range(1, len(y_train) + 1))
        x_pre_initial = list(range(len(y_train) + 1, len(y_train) + 2 + self.forecast_days + 1))
        min_accuracy = 10.0
        min_y_pre = []
        for degree in range(2, 100):
            # print(degree)
            x_train = data_ascension(x_initial, degree)
            x_pre = data_ascension(x_pre_initial, degree)
            model.fit(x_train, y_train)
            y_pre_initial = model.predict(x_pre)
            y_test_pre = y_pre_initial[:2]
            y_pre = y_pre_initial[2:]
            accuracy = value_accuracy(y_test_pre, y_test)
            if accuracy <= min_accuracy:
                min_accuracy = accuracy
                min_y_pre = y_pre
            # print(accuracy)
            if accuracy <= 0.1:
                result = dict.fromkeys(self.return_list)
                result['y_predict'] = y_pre.tolist()
                result['is_correct'] = True
                result['predict_accuracy'] = accuracy
                result['model_name'] = self.use_model_name.name
                return json.dumps(result)
        result = dict.fromkeys(self.return_list)
        result['y_predict'] = min_y_pre.tolist()
        result['is_correct'] = False
        result['predict_accuracy'] = min_accuracy
        result['model_name'] = self.use_model_name.name
        return json.dumps(result)

    def train_model_complex(self, y_train_initial: list):
        """
        复杂回归分析。

        使用3天的数据预测下一天的数据，精度较简单回归分析有明显提高。

        返回值说明
        ----
        - y_predict：Y值的预测列表
        - is_correct：布尔值，该算法只迭代一轮，该值无意义，恒为True。
        - predict_accuracy 该值恒为-1，用于标记。
        - model_name 使用的模型名称
        :param y_train_initial: y_train_initial: Y值列表
        :return: JSON列表
        """
        model = self.select_model(self.use_model_name)
        x_test = []
        y_test = y_train_initial[3:]
        x_pre_initial = y_train_initial
        y_pre = []
        for i in range(len(y_train_initial) - 3):
            x_test_once = [y_train_initial[i], y_train_initial[i + 1], y_train_initial[i + 2]]
            x_test.append(x_test_once)
        model.fit(x_test, y_test)
        for i in range(self.forecast_days):
            x_pre_once = [[x_pre_initial[-3], x_pre_initial[-2], x_pre_initial[-1]]]
            y_pre_once = model.predict(x_pre_once)[0]
            y_pre.append(y_pre_once)
            x_pre_initial.append(y_pre_once)
        result = dict.fromkeys(self.return_list)
        result['y_predict'] = y_pre
        result['is_correct'] = True
        result['predict_accuracy'] = -1
        result['model_name'] = self.use_model_name.name
        return json.dumps(result)


def bill_predict(y_real: list, is_complex_model: bool = False, forecast_days: int = 3,
                 use_model_name='linear_regression'):
    """
    API接口函数

    参数说明：
    ----
    枚举值（模型名称可填）：

    - bayesian_ridge 贝叶斯岭回归模型
    - linear_regression 线性回归模型
    - elastic_net 弹性网络回归模型
    - svr 向量机回归模型
    - gradient_boosting_regressor 梯度增强回归模型

    返回值说明：
    ----
    - y_predict：Y值的预测列表
    - is_correct：布尔值，是否准确。
    - predict_accuracy：准确率
    - model_name：使用的模型名称
    :param y_real:（必填） Y值列表
    :param is_complex_model:（可选） 布尔值，是否使用复杂模型，True为使用，False为使用简单模型，默认为False
    :param forecast_days:（可选） 预测天数，默认为3天
    :param use_model_name:（可选） 使用模型名称，默认为线性回归模型
    :return: JSON列表
    """
    rp = RegressionPrediction(forecast_days=forecast_days, use_model_name=use_model_name)
    if is_complex_model:
        return rp.train_model_complex(y_real)
    else:
        return rp.train_model_easy(y_real)


if __name__ == '__main__':
    # rep = RegressionPrediction(forecast_days=4, use_model_name='elastic_net')
    y = [12, 23, 32, 14, 53]
    y_p = bill_predict(y, is_complex_model=True, )
    print(y_p)
