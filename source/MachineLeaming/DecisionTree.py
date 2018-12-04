from math import log
import operator
import matplotlib.pyplot as plt


def create_data_set():
    data_set = [
        [1, 1, 'Y'],
        [1, 1, 'Y'],
        [1, 0, 'N'],
        [0, 1, 'N'],
        [0, 1, 'N'],
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def majority_count(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 1
        else:
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def calc_shannon_ent(data_set):
    num_entries = len(data_set)
    label_map = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_map.keys():
            label_map[current_label] = 1
        else:
            label_map[current_label] += 1
    shannon_ent = 0
    for count in label_map.values():
        prob = count / num_entries
        shannon_ent -= prob * log(prob,2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    new_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            new_data_set.append(reduced_feat_vec)
    return new_data_set


def choose_best_split_feature(data_set):
    feature_count = len(data_set[0]) - 1
    base_ent = calc_shannon_ent(data_set)
    best_info_gain = 0
    best_feature = -1
    for axis in range(feature_count):
        features = [ele[axis] for ele in data_set]
        unique_values = set(features)
        new_ent = 0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, axis, value)
            prob = len(sub_data_set)/len(data_set)
            new_ent += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = axis
    return best_feature


def create_tree(data_set, labels):
    class_list = [ele[-1] for ele in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set) == 1:
        return majority_count(class_list)

    best_feature = choose_best_split_feature(data_set)
    best_feature_label = labels[best_feature]
    node = {
        best_feature_label:{},
    }
    del(labels[best_feature])
    feature_values = [ele[best_feature] for ele in data_set]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_labels = labels[:]
        node[best_feature_label][value] = create_tree(split_data_set(data_set,best_feature,value),sub_labels)
    return node


def draw_tree(tree):
    decision_node = dict(boxstyle="sawtooth", fc="0.8")
    leaf_node = dict(boxstyle="round4",fc="0.8")
    arrow_arg = dict(arrowstyle="<-")

    def plot_node(node_text, center_point, parent_point, node_type):
        create_plot.ax1.annotate(node_text,
                                 xy=parent_point,
                                 xycoords='axes fraction',
                                 xytext=center_point,
                                 textcoords='axes fraction',
                                 va="center", ha="center",
                                 bbox=node_type,
                                 arrowprops=arrow_arg)

    def plot_mid_text(center_point, parent_point, text):
        x = (parent_point[0] - center_point[0])/2 + center_point[0]
        y = (parent_point[1] - center_point[1])/2 + center_point[1]
        create_plot.ax1.text(x, y, text)

    def plot_tree(sub_tree, parent_point, node_text):
        leafs_count = get_leafs_count(sub_tree)
        tree_type = list(sub_tree.keys())[0]
        tree_node = sub_tree[tree_type]

        center_point = (plot_tree.xOff + (1.0 + float(leafs_count))/2.0/plot_tree.totalW, plot_tree.yOff)
        tree_node = sub_tree[tree_type]
        plot_mid_text(center_point, parent_point, node_text)
        plot_node(tree_type, center_point, parent_point, decision_node)

        plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD
        for key in tree_node:
            if type(tree_node[key]).__name__ == 'dict':
                plot_tree(tree_node[key], center_point, str(key))
            else:
                plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
                plot_node(tree_node[key], (plot_tree.xOff, plot_tree.yOff), center_point, leaf_node)
                plot_mid_text((plot_tree.xOff, plot_tree.yOff), center_point, str(key))
        plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD

    def create_plot():
        fig = plt.figure(1, facecolor="white")
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
        plot_tree.totalW = float(get_leafs_count(tree))
        plot_tree.totalD = float(get_tree_depth(tree))
        plot_tree.xOff = -0.5/plot_tree.totalW
        plot_tree.yOff = 1
        plot_tree(tree, (0.5, 1), '')
        plt.show()

    def get_leafs_count(sub_tree):
        count = 0
        for node in sub_tree.values():
            if type(node).__name__ == 'dict':
                count += get_leafs_count(node)
            else:
                count += 1
        return count

    def get_tree_depth(sub_tree, depth=-1):
        for node in sub_tree.values():
            if type(node).__name__ == 'dict':
                depth = get_tree_depth(node, depth)
            else:
                depth += 1
        return depth

    leafs_count = get_leafs_count(tree)
    tree_depth = get_tree_depth(tree)
    print(leafs_count,tree_depth)
    create_plot()


if __name__ == '__main__':
    ds, lab = create_data_set()
    the_tree = create_tree(ds,lab)
    print(ds, the_tree)
    draw_tree(the_tree)
