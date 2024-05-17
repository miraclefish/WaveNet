import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps
from matplotlib.lines import Line2D
import numpy as np
import networkx as nx

COLUMNS2INDEX = {'FP1-F7': 0,
                 'F7-T3': 1,
                 'T3-T5': 2,
                 'T5-O1': 3,
                 'FP2-F8': 4,
                 'F8-T4': 5,
                 'T4-T6': 6,
                 'T6-O2': 7,
                 'A1-T3': 8,
                 'T3-C3': 9,
                 'C3-CZ': 10,
                 'CZ-C4': 11,
                 'C4-T4': 12,
                 'T4-A2': 13,
                 'FP1-F3': 14,
                 'F3-C3': 15,
                 'C3-P3': 16,
                 'P3-O1': 17,
                 'FP2-F4': 18,
                 'F4-C4': 19,
                 'C4-P4': 20,
                 'P4-O2': 21,
                 'EKG': 22}

ARTIFACT_TO_ID = {
    'null': 0,
    'eyem': 1,
    'chew': 2,
    'shiv': 3,
    'musc': 4,
    'elpp': 5,
    'elec': 6,
    'eyem_chew': 7,
    'eyem_shiv': 8,
    'eyem_musc': 9,
    'eyem_elec': 10,
    'chew_musc': 11,
    'chew_elec': 12,
    'shiv_elec': 13,
    'musc_elec': 14
}

def PlotEEGMontage(eeg_signal, time, length, label=None, pred=None, file_name=None, Sens=1.5, save_fig=None, sfreq=125):
    L, N = eeg_signal.shape
    # sfreq = 250

    start, end, ll = adjust_window(time, length, L, sfreq)

    columns = eeg_signal.columns

    data = np.clip(eeg_signal.values, -500, 500)
    x_index = np.linspace(time, time + ll, end - start)

    anchor_xaxis = np.arange(0, N * 100 * Sens, 100 * Sens)

    fig = plt.figure(figsize=[15, 10])
    ax = fig.add_subplot(1, 1, 1)


    cmap = plt.get_cmap("Set2")

    colors = [cmap(i) for i in np.linspace(0, 1, 23)]
    if label is not None:

        label = label[(label['start'] < end) & (label['end'] > start)]
        label.loc[label['end'] > end, 'end'] = end
        label.loc[label['start'] < start, 'start'] = start
        ID_TO_ARTIFACT = {v: k for k, v in ARTIFACT_TO_ID.items()}

        for label_start, label_end, j_list, k in label.loc[:, ['start', 'end', '#Channel', 'label']].values:
            for j in j_list:
                rect = plt.Rectangle((x_index[label_start - start], (N - j - 1 - 0.5) * 100 * Sens),
                                     (label_end - label_start) / sfreq, 100 * Sens * 0.8)
                rect.set(facecolor='gray', alpha=0.5)
                ax.add_patch(rect)
                ax.text(x_index[label_start - start], (N - j - 1) * 100 * Sens, ID_TO_ARTIFACT[k], color='black', fontsize=14)

    if pred is not None:

        pred = pred[(pred['start'] < end) & (pred['end'] > start)]
        new_pred = pred.copy()
        new_pred[['start', 'end']] = pred[['start', 'end']].clip(start, end)

        if 'label' not in pred.columns:
            for pred_start, pred_end, j in new_pred.loc[:, ['start', 'end', '#Channel']].values:
                line = plt.plot(x_index[pred_start - start:pred_end - start],
                                data[pred_start:pred_end, j] + anchor_xaxis[N - j - 1],
                                color='red',
                                linewidth=5)
                # ax.text(x_index[pred_start - start], (N - j - 1.5) * 100 * Sens, name, color=pred_colors[k])

            # 创建自定义的 Line2D 对象
            lines = []
            lines.append(Line2D([], [], color='red', linestyle='-', label='Artifacts'))
        else:
            colors = ['black'] * 23
            pred_colors = ['white', 'red', 'skyblue', 'limegreen', 'pink', 'gold']
            linewidths = [1, 2, 3, 4, 5, 6]
            label_name = ['eyem', 'chew', 'shiv', 'musc', 'elec']
            new_pred = new_pred.sort_values('label', ascending=False)
            for pred_start, pred_end, j, k in new_pred.loc[:, ['start','end', '#Channel', 'label']].values:
                line = plt.plot(x_index[pred_start - start:pred_end - start],
                                data[pred_start:pred_end, j] + anchor_xaxis[N - j - 1],
                                color=pred_colors[k],
                                linewidth=linewidths[k])
                # ax.text(x_index[pred_start - start], (N - j - 1.5) * 100 * Sens, label_name[k-1], color=pred_colors[k])

            # 创建自定义的 Line2D 对象
            lines = []
            for c, n in zip(pred_colors[1:], label_name):
                lines.append(Line2D([], [], color=c, linestyle='-', label=n))

        # 创建图例
        legend = plt.legend(handles=lines, loc='upper right', fontsize=14)

        ax.add_artist(legend)

    for i in range(N):
        ax.plot(x_index, data[start:end, N - i - 1] + anchor_xaxis[i], color=colors[N-i-1], linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(anchor_xaxis)
    ax.set_yticklabels(columns[::-1], rotation=0, fontsize=14)
    ax.set_ylim([-100 * Sens, 100 * Sens * N])
    ax.set_xticks(list(np.arange(time, time + length + 1e-6, length / 5)))
    ax.tick_params(axis='x', labelsize=14)

    plt.title('[{}] {}s-{}s'.format(file_name, time, time + length), fontsize=16)
    # plt.subplots_adjust(hspace=0.05)
    if save_fig is not None:
        path = "{}/{}_{}.png".format(save_fig, file_name, time)
        # path = os.path.join('PlotDataset2', "{}-{}.png".format(save_fig, time))
        plt.savefig(path, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return None

def draw_binary_tree(root, max_depth, file_name, time, channel, save_fig=None):
    G = nx.Graph()
    pos = {}

    add_edges(G, root, 1, max_depth, pos, layer_width=3)

    node_values = list(G.nodes)
    node_colors = [float(value) for value in node_values]
    node_labels = {node: data['label'] for node, data in G.nodes(data=True)}

    pos_attrs = nx.get_node_attributes(G, 'pos')
    cmap = colormaps['Reds']  # 使用Blues colormap
    fig = plt.figure()
    nx.draw(G, pos=pos_attrs, with_labels=True, labels=node_labels, node_size=700, node_color=node_colors, cmap=cmap, font_size=8,
            font_color='black')

    c = {v: k for k, v in COLUMNS2INDEX.items()}

    plt.title("{} {}s-{}s Wavelet_Tree of Channel {}".format(file_name, time, time+5, c[channel-1]))

    # 添加颜色条
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_values), vmax=max(node_values)))
    # sm.set_array([])

    # 使用 mappable 参数添加颜色条
    # cbar = plt.colorbar(sm, orientation='vertical', pad=0.05, aspect=50, mappable=sm)
    # cbar.set_label('Node Values')
    if save_fig is not None:
        path = "{}_{}_{}_Wavelet_Tree_{}.png".format(save_fig, file_name, time, channel)
        # path = os.path.join('PlotDataset2', "{}-{}.png".format(save_fig, time))
        plt.savefig(path, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return None

def add_edges(graph, parent_node, depth, max_depth, pos=None, x=0, y=0, layer_width=1):
    if depth <= max_depth:
        current_node = (x, y)
        if len(graph.nodes) == 0:
            graph.add_node(parent_node.value, pos=current_node, label=f"{parent_node.value.round(2)}")

        left_child_node = parent_node.left
        right_child_node = parent_node.right


        if left_child_node:
            left_child_name = f"{left_child_node.value.round(2)}"
            left_child_pos = (x - layer_width / 2, y - 1)
            while left_child_node.value in graph.nodes:
                left_child_node.value += 1e-6
            graph.add_node(left_child_node.value, pos=left_child_pos, label=left_child_name)
            graph.add_edge(parent_node.value, left_child_node.value)

            add_edges(graph, left_child_node, depth + 1, max_depth, pos=left_child_pos, x=left_child_pos[0], y=left_child_pos[1],
                      layer_width=layer_width / 2)

        if right_child_node:
            right_child_name = f"{right_child_node.value.round(2)}"
            right_child_pos = (x + layer_width / 2, y - 1)
            while right_child_node.value in graph.nodes:
                right_child_node.value += 1e-6
            graph.add_node(right_child_node.value, pos=right_child_pos, label=right_child_name)
            graph.add_edge(parent_node.value, right_child_node.value)

            add_edges(graph, right_child_node, depth + 1, max_depth, pos=right_child_pos, x=right_child_pos[0], y=right_child_pos[1],
                      layer_width=layer_width / 2)

def draw_sub_bands(tree, bands, time_index, label, pred, file_name, channel, save_fig=None):

    n = len(tree)
    start = time_index * 625

    fig = plt.figure(figsize=[12, 12])
    for i, (w, band) in enumerate(zip(tree, bands)):
        ax = plt.subplot(n, 1, i+1)
        x = band[w.argmax()]

        id = w.argmax()
        if i == 0:
            title = "{} {}s-{}s Subbands of Channel {}".format(file_name, time_index*5, time_index*5+5, channel)
        else:
            component = find_id_and_level(id, i)
            title = 'Sub-band {} in level {}: Original'.format(id+1, i) + component

        plt.plot(x)
        if i == 0:
            if len(label) > 0:
                for label_start, label_end, name in label.loc[:, ['start', 'end', 'label']].values:
                    rect = Rectangle((label_start - start, min(x)), label_end - label_start, max(x) - min(x),
                                     facecolor='gray', alpha=0.5, label='Ground Truth')
                    ax.add_patch(rect)
                    ax.text(label_start - start, min(x), name, color='black')
        if len(pred) > 0:
            for pred_start, pred_end in pred.loc[:, ['start', 'end']].values:
                s = (pred_start - start) // (2 ** i)
                e = (pred_end - start) // (2 ** i)
                plt.plot(np.arange(s, e), x[s:e], color='red', linewidth=2, label='Prediction')
                # ax.text(x_index[pred_start - start], (N - j - 1.5) * 100 * Sens, name, color=pred_colors[k])

                if i == 0:
                    ax.legend()
        plt.title(title)

    plt.tight_layout()

    if save_fig is not None:
        path = "{}_{}_{}_Subbands_{}.png".format(save_fig, file_name, time_index*5, channel)
        # path = os.path.join('PlotDataset2', "{}-{}.png".format(save_fig, time))
        plt.savefig(path, dpi=500, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return None


def find_id_and_level(id, level):
    component = ""
    for i in range(level):
        if (id+1) % 2 == 0:
            component = '->High' + component
        else:
            component = '->Low' + component
        id = id // 2
    return component



def adjust_window(time, length, L, sfreq):
    time = int(time * sfreq)
    length = int(length * sfreq)
    start = time
    end = time + length

    if start < 0:
        start = 0
    elif end > L - 1:
        end = L - 1
    elif start > L - 1:
        start = L - 1 - length
        end = L - 1
    ll = (end - start) / sfreq

    return start, end, ll