import numpy as np

'''

'''

# 数据集
def load_dataset(filename):
    dataset = []
    file = open(filename)
    for line in file.readlines():
        current = line.split(',')
        value = list(map(float, current))
        dataset.append(value)
    return np.mat(dataset)

# 计算总的方差
# np.var
def calc_variance(dataset):
    return np.var(dataset[:, -1])


# 根据给定的特征、特征值划分数据集
# numpy.nonzero(a): 返回的是a中非0元素的索引的元组
# (array([0, 1]), array([3, 5])) 代表(0,3),(1,5)位置的元素非0
def dataset_split(dataset,feature_axis,feature_value):
    # 取出dataset指定特征列(feature_axis)中的非0元素，并且该元素大于特征值(feature_value)
    dataset_left = dataset[np.nonzero(dataset[:, feature_axis] > feature_value)[0], :]
    dataset_right = dataset[np.nonzero(dataset[:, feature_axis] <= feature_value)[0], :]
    return dataset_left, dataset_right


# 特征划分
# threshold_variance: 剪枝前总方差与剪枝后总方差差值的最小值
# branch_data_limit: 子数据集中的样本的最少数量
# 后两个参数提供预剪枝功能

def chose_best_feature(dataset, threshold_variance, branch_data_limit):
    feature_values = dataset[:, -1]
    # 数据集只有一条数据
    if feature_values.shape[0] == 1:
        # print("feature values shape", feature_values.shape)
        return None, np.mean(feature_values)

    dataset_variance = calc_variance(dataset)
    best_feature_anix = -1
    best_value = 0
    minimum_variance = np.inf

    feature_count = dataset.shape[1]
    for axis in range(feature_count-1):
        # 将指定列所有数据转为list
        values = dataset[:, axis].T.tolist()[0]
        # 排除重复项,转为set
        values = set(values)
        for value in values:
            dataset_left, dataset_right = dataset_split(dataset, axis, value)
            # 判断切分之后数量
            if dataset_left.shape[0] < branch_data_limit or dataset_right.shape[0] < branch_data_limit:
                # print("split dataset axis(%d), split(%f)" % (axis, value))
                continue
            # 计算方差
            temp_variance = calc_variance(dataset_left) + calc_variance(dataset_right)
            if temp_variance < minimum_variance:
                minimum_variance = temp_variance
                best_value = value
                best_feature_anix = axis

    # 总体方差和最小方差差距小于阈值
    if abs(dataset_variance - minimum_variance) < threshold_variance:
        # print("dataset variance %f, minimum variance %f" % (dataset_variance, minimum_variance))
        return None, np.mean(feature_values)

    dataset_left, dataset_right = dataset_split(dataset, best_feature_anix, best_value)
    # 判断切分之后数量
    if dataset_left.shape[0] < branch_data_limit or dataset_right.shape[0] < branch_data_limit:
        return None, np.mean(feature_values)
    return best_feature_anix, best_value


# 用于判断所给的节点是否是叶子节点
def is_tree(node):
    return (type(node).__name__=='dict' )


# 计算两个叶子节点的均值
def get_tree_mean(node):
    leftmean,rightmean = 0, 0
    if is_tree(node['left']):
        leftmean = get_tree_mean(node['left'])
    if is_tree(node['right']):
        rightmean = get_tree_mean(node['right'])
    return (leftmean+rightmean)/2.0


def create_tree(dataset, threshold_variance=0, branch_data_limit=1):
    feature_axis,feature_value = chose_best_feature(dataset, threshold_variance, branch_data_limit)
    if feature_axis is None:
        return feature_value

    tree = {
        'axis': feature_axis,
        'split': feature_value,
    }
    dataset_left, dataset_right = dataset_split(dataset, feature_axis, feature_value)

    tree['left'] = create_tree(dataset_left, threshold_variance, branch_data_limit)
    tree['right'] = create_tree(dataset_right,  threshold_variance, branch_data_limit)

    return tree


# 后剪枝
def prune_tree(tree, testdata):
    pass

# 预测
def forecast(tree,testdata):
    if not is_tree(tree): return float(tree)
    # print"选择的特征是：" ,Tree['spInd']
    # print"测试数据的特征值是：" ,testData[Tree['spInd']]
    if testdata[0,tree['axis']] > tree['split']:
        if is_tree(tree['left']):
            return forecast(tree['left'],testdata)
        else:
            return float(tree['left'])
    else:
        if is_tree(tree['right']):
            return forecast(tree['right'],testdata)
        else:
            return float(tree['right'])


def TreeForecast(tree, testData):
    m = np.shape(testData)[0]
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[i,0] = forecast(tree, testData[i])
    return y_hat


if __name__ == '__main__':
    ds_train = load_dataset("../../resource/CART_train.txt")
    ds_test = load_dataset("../../resource/CART_test.txt")
    tree = create_tree(ds_train, 1)
    y = ds_test[:, -1]
    y_hat = TreeForecast(tree, ds_test)
    print(tree,np.corrcoef(y_hat, y, rowvar=0)[0, 1])
