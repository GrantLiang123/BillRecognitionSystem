"""
该模块实现了基于神经网络的错误数据识别方法。通过生成包含正常数据和异常值
的二维数据集，并对数据进行标准化处理。然后，构建了一个多层感知器（MLP）神经网络模型，
包括多个隐藏层和输出层，用于对数据进行分类，判断数据点是否为异常值。
模型训练采用二元交叉熵损失函数和Adam优化器，在训练过程中通过学习率调度器
动态调整学习率。最后（测试中使用），评估模型的性能并可视化识别结果，以实现对数据中错误值的有效识别和检测。
"""

# coding : utf-8
# Time : 2024/6/3
# Author : Liang Zi Fan
# File : regression_prediction.py
# Software : Pycharm

import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_data_test():
    # 数据准备
    train_data_length = 1024
    train_data = torch.zeros((train_data_length, 2))
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)  # x
    train_data[:, 1] = torch.sin(train_data[:, 0])  # y

    # 添加异常值
    num_outliers = 50
    outliers = torch.zeros((num_outliers, 2))
    outliers[:, 0] = 2 * math.pi * torch.rand(num_outliers)
    outliers[:, 1] = 2 * (torch.rand(num_outliers) - 0.5)  # 随机噪声作为异常值

    # 合并正常数据和异常值
    train_data = torch.cat([train_data, outliers], dim=0)
    train_labels = torch.cat([torch.zeros(train_data_length), torch.ones(num_outliers)], dim=0)  # 0表示正常数据，1表示异常值

    # 添加标志位，标记手动添加的异常值
    flag_normal = torch.zeros(train_data_length, 1)
    flag_outliers = torch.ones(num_outliers, 1)
    flags = torch.cat([flag_normal, flag_outliers], dim=0)

    # 数据标准化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)

    # 创建数据集和数据加载器
    dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return train_data, train_labels, train_loader, flags


def prepare_data(data_list):
    train_data_length = len(data_list)
    train_data = torch.zeros((train_data_length, 2))
    for i in range(train_data_length):
        train_data[i, 0] = i  # x
        train_data[i, 1] = data_list[i]  # y

    data_border = max(data_list) + min(data_list)
    # print(data_border)

    # 添加异常值
    num_outliers = int(len(data_list) / 20)
    outliers = torch.zeros((num_outliers, 2))
    # print(len(data_list) + 1)
    outliers[:, 0] = (len(data_list) + 1) * torch.rand(num_outliers)
    outliers[:, 1] = data_border * torch.rand(num_outliers)  # 随机噪声作为异常值
    # print(outliers)

    # 合并正常数据和异常值
    train_data = torch.cat([train_data, outliers], dim=0)
    train_labels = torch.cat([torch.zeros(train_data_length), torch.ones(num_outliers)], dim=0)  # 0表示正常数据，1表示异常值

    # 添加标志位，标记手动添加的异常值
    flag_normal = torch.zeros(train_data_length, 1)
    flag_outliers = torch.ones(num_outliers, 1)
    flags = torch.cat([flag_normal, flag_outliers], dim=0)

    # 数据标准化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)

    # 创建数据集和数据加载器
    dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return train_data, train_labels, train_loader, flags


# 定义神经网络模型
class ImprovedNN(nn.Module):
    def __init__(self):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x


def build_model(data_list):
    """
    预测错误数据。

    返回值形式如下：
    一个<class 'numpy.ndarray'>类型的二维列表，第一列是错误数据在data_list
    中的索引号（从零开始）；第二个是错误数据的值。全部使用float32形式储存，
    注意截断误差，数据值可能与原值不一致。

    [[index,value],
     [index,value],
     ...
     [index,value]]

    :param data_list: 包含按时间顺序排列的一维列表
    :return: output_data_errors
    """
    train_data_length = len(data_list)
    train_data = torch.zeros((train_data_length, 2))
    for i in range(train_data_length):
        train_data[i, 0] = i  # x
        train_data[i, 1] = data_list[i]  # y

    data_border = max(data_list) + min(data_list)
    # print(data_border)

    # 添加异常值
    num_outliers = int(len(data_list) / 20)
    outliers = torch.zeros((num_outliers, 2))
    # print(len(data_list) + 1)
    outliers[:, 0] = (len(data_list) + 1) * torch.rand(num_outliers)
    outliers[:, 1] = data_border * torch.rand(num_outliers)  # 随机噪声作为异常值
    # print(outliers)

    # 合并正常数据和异常值
    train_data = torch.cat([train_data, outliers], dim=0)
    train_labels = torch.cat([torch.zeros(train_data_length), torch.ones(num_outliers)], dim=0)  # 0表示正常数据，1表示异常值

    # 添加标志位，标记手动添加的异常值
    flag_normal = torch.zeros(train_data_length, 1)
    flag_outliers = torch.ones(num_outliers, 1)
    flags = torch.cat([flag_normal, flag_outliers], dim=0)

    # 数据标准化
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)

    # 创建数据集和数据加载器
    dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ImprovedNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练模型
    num_epochs = 200  # 训练次数

    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

        scheduler.step()  # 更新学习率

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 评估模型性能
    model.eval()
    with torch.no_grad():
        predictions = model(train_data).squeeze()
        predicted_labels = (predictions >= 0.15).float()
        accuracy = (predicted_labels == train_labels).sum().item() / train_labels.size(0)
        print(f'Accuracy: {accuracy:.4f}')

    # 找出被模型标记为错误的数据点
    outliers_detected = train_data[predicted_labels == 1]
    detected_flags = flags[predicted_labels == 1]
    print(outliers_detected)
    print(len(outliers_detected))
    # output = outliers_detected[detected_flags.squeeze() == 0]
    # print(output)
    # print(len(output))
    # 排除手动添加的异常值
    original_data_errors = outliers_detected[detected_flags.squeeze() == 0]

    # 将检测出的原始数据中的错误值还原为原始尺度
    original_data_errors = original_data_errors.numpy()
    output_data_errors = scaler.inverse_transform(original_data_errors)
    print(output_data_errors)
    print(len(output_data_errors))
    print(type(output_data_errors))

    # 可视化检测结果
    plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, cmap='coolwarm', label='Data')
    # print(train_data)
    plt.scatter(outliers_detected[:, 0], outliers_detected[:, 1], edgecolor='k', facecolors='none',
                label='Detected Outliers')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    return output_data_errors


if __name__ == '__main__':
    data = []
    for i in range(1000):
        data.append(i)
    # print(data)
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
            83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
            108, 109, 110, 111, 112, 113, 34, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
            129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
            150, 151, 152, 153, 154, 155, 156, 157, 743, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
            170,
            171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
            192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
            213, 214, 215, 216, 217, 218, 219, 220, 923, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232,
            233,
            234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
            255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
            276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 865, 288, 289, 290, 291, 292, 293, 294, 295,
            296,
            297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317,
            318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
            339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
            360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380,
            381, 382, 383, 384, 385, 386, 32, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401,
            402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422,
            423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443,
            444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 5, 457, 458, 459, 460, 461, 462, 463, 464,
            465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485,
            486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
            507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
            528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 46, 543, 544, 545, 546, 547, 548,
            549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569,
            570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590,
            591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 723, 604, 605, 606, 607, 608, 609, 610,
            611,
            612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632,
            633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653,
            654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674,
            675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695,
            696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716,
            717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 45, 732, 733, 734, 735, 736, 737,
            738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758,
            759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
            780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800,
            801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821,
            822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842,
            843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863,
            864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884,
            885, 886, 887, 888, 889, 890, 891, 892, 893, 1, 894, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 905,
            906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926,
            927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947,
            948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968,
            969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989,
            990, 991, 992, 993, 994, 995, 996, 997, 998, 999]
    # print(data)
    # my_train_data, my_train_labels, my_train_loader, my_flags = prepare_data(data)
    # my_train_data, my_train_labels, my_train_loader, my_flags = prepare_data_test()
    # print(my_train_data)
    build_model(data)
