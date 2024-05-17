import numpy as np
import pandas as pd
from tqdm import tqdm

SINGLE_LABEL_TO_MULTI_LABEL = {
    0: [0,],
    1: [1,],
    2: [2,],
    3: [3,],
    4: [4,],
    5: [5,],
    6: [5,],
    7: [1, 2],
    8: [1, 3],
    9: [1, 4],
    10: [1, 5],
    11: [2, 4],
    12: [2, 5],
    13: [3, 5],
    14: [4, 5]
}

SINGLE_LABEL_TO_BINARY_LABEL = {i: [0,] for i in range(15)}

def log_metrics(logger, metric_name, values):
    if len(values) > 1:
        v_list = []
        for k, v in values.items():
            logger.run[f'{metric_name}/{k}'] = v
            v_list.append(v)
        avg_key_name = k.split('/')[0] + '/Avg'
        logger.run[f'{metric_name}/{avg_key_name}'] = np.mean(v_list)
    else:
        for k, v in values.items():
            logger.run[f'{metric_name}/{k}'] = v
    return None

class BASEevaluator:

    def __init__(self, prediction, base_label, task='multi_cls', th=0.5, logger=None):

        self.prediction = prediction
        self.label = base_label
        self.task = task
        self.th = th
        self.logger = logger

    def get_metrics(self):

        TP, FP, FN = self.get_all_confusion_matrix()

        if self.task == 'multi_cls':
            LABEL_NAMES = ['eyem', 'chew', 'shiv', 'musc', 'elpp+elec']
            TP = TP[:, 1:]
            FP = FP[:, 1:]
            FN = FN[:, 1:]

        if self.task == 'binary_cls':
            LABEL_NAMES = ['Artifacts']

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        precision = {f'Pre/{label}': value for label, value in zip(LABEL_NAMES, precision[0])}
        recall = {f'Rec/{label}': value for label, value in zip(LABEL_NAMES, recall[0])}
        f1_score = {f'F1/{label}': value for label, value in zip(LABEL_NAMES, f1_score[0])}

        if self.logger:
            log_metrics(logger=self.logger, metric_name='BASE', values=precision)
            log_metrics(logger=self.logger, metric_name='BASE', values=recall)
            log_metrics(logger=self.logger, metric_name='BASE', values=f1_score)

        return precision, recall, f1_score

    def get_all_confusion_matrix(self):

        num_class = self.prediction[0].shape[-1]
        TP_all, FP_all, FN_all = np.zeros((1, num_class)), np.zeros((1, num_class)), np.zeros((1, num_class))
        with tqdm(total=len(self.prediction)) as pbar:
            pbar.set_description('BASE metrics evaluating: ')
            for i in range(len(self.prediction)):
                prediction = self.prediction[i]
                label = self.label[i]
                TP_subject, FP_subject, FN_subject = self.compute_confusion_matrix(prediction, label)
                TP_all += TP_subject
                FP_all += FP_subject
                FN_all += FN_subject
                pbar.update(1)

        return TP_all, FP_all, FN_all

    def compute_confusion_matrix(self, prediction, label):

        '''
        prediction: L x C x num_class
        label: L x C x num_class
        return:
            TP: 1 x num_class
            FN: 1 x num_class
            FP: 1 x num_class
        '''

        assert prediction.shape == label.shape, "Error: the shapes of prediction and label are different."

        y_pred = prediction > self.th
        TP = (y_pred * label).sum(axis=0).sum(axis=0, keepdims=True)
        FP = y_pred.sum(axis=0).sum(axis=0, keepdims=True) - TP
        FN = label.sum(axis=0).sum(axis=0, keepdims=True) - TP
        return TP, FP, FN


class EACSevaluator:

    def __init__(self, prediction, event_label, task: str = 'multi_cls', th: float = 0.5, min_l: int = 25, logger=None):

        self.prediction = prediction
        self.label = event_label
        self.task = task
        self.th = th
        self.min_l = min_l
        self.logger = logger

    def get_metrics(self):

        TP, FP, FN = self.get_all_confusion_matrix()

        if self.task == 'multi_cls':
            LABEL_NAMES = ['eyem', 'chew', 'shiv', 'musc', 'elpp+elec']
            TP = TP[1:]
            FP = FP[1:]
            FN = FN[1:]

        if self.task == 'binary_cls':
            LABEL_NAMES = ['Artifacts']

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        precision = {f'Pre/{label}': value for label, value in zip(LABEL_NAMES, precision)}
        recall = {f'Rec/{label}': value for label, value in zip(LABEL_NAMES, recall)}
        f1_score = {f'F1/{label}': value for label, value in zip(LABEL_NAMES, f1_score)}

        if self.logger:
            log_metrics(logger=self.logger, metric_name='EACS', values=precision)
            log_metrics(logger=self.logger, metric_name='EACS', values=recall)
            log_metrics(logger=self.logger, metric_name='EACS', values=f1_score)

        return precision, recall, f1_score

    def get_all_confusion_matrix(self):

        num_class = self.prediction[0].shape[-1]
        TP_all, FP_all, FN_all = np.zeros(num_class), np.zeros(num_class), np.zeros(num_class)
        with tqdm(total=len(self.prediction)) as pbar:
            pbar.set_description('EACS metrics evaluating: ')
            for i in range(len(self.prediction)):
                prediction = self.prediction[i]
                prediction = self.transfer_prediction(prediction)
                label = self.label[i]
                TP_subject, FP_subject, FN_subject = self.compute_confusion_matrix(prediction, label)
                TP_all += TP_subject
                FP_all += FP_subject
                FN_all += FN_subject
                pbar.update(1)

        return TP_all, FP_all, FN_all

    def compute_confusion_matrix(self, prediction, label):

        if self.task == 'multi_cls':
            label['label'] = label['label'].apply(lambda x: SINGLE_LABEL_TO_MULTI_LABEL[x])
            start_class_id = 1
        if self.task == 'binary_cls':
            label['label'] = label['label'].apply(lambda x: SINGLE_LABEL_TO_BINARY_LABEL[x])
            start_class_id = 0

        num_class = self.prediction[0].shape[-1]
        TP, FP, FN = np.zeros(num_class), np.zeros(num_class), np.zeros(num_class)
        for i in range(start_class_id, num_class):
            prediction_class_i = prediction[prediction['label'] == i]
            label_class_i = label[label['label'].apply(lambda x: True if i in x else False)]
            tp, fp, fn = self.compute_single_class_confusion_matrix(prediction_class_i, label_class_i)
            TP[i] += tp
            FP[i] += fp
            FN[i] += fn

        return TP, FP, FN

    def compute_single_class_confusion_matrix(self, prediction, label):

        TP, FP, FN = 0, 0, 0
        for i in range(22):
            prediction_channel_i = prediction[prediction['#Channel'] == i]
            label_channel_i = label[label['#Channel'].apply(lambda x: True if i in x else False)]
            tp, fp, fn = self.compute_single_class_channel_confusion_matrix(prediction_channel_i, label_channel_i)
            TP += tp
            FP += fp
            FN += fn
        return TP, FP, FN

    def compute_single_class_channel_confusion_matrix(self, prediction, label):

        overlap_flag = None
        tp = 0
        fp = 0
        fn = 0

        if label.shape[0] == 0:
            fp +=prediction.shape[0]
            return tp, fp, fn
        else:
            for anchor_s, anchor_e in label.loc[:, ['start', 'end']].values:
                overlap = (prediction['start'] < anchor_e) & (prediction['end'] > anchor_s)
                if overlap_flag is None:
                    overlap_flag = overlap
                else:
                    overlap = (~ overlap_flag) & overlap
                    overlap_flag = overlap_flag | overlap
                prediction_overlap_anchor = prediction[overlap]

                fn_anchor = 1
                for pred_s, pred_e in prediction_overlap_anchor.loc[:, ['start', 'end']].values:
                    overlap_segment = min(pred_e, anchor_e) - max(pred_s, anchor_s)
                    pred_length = pred_e - pred_s
                    anchor_length = anchor_e - anchor_s
                    error_pred_segment = pred_length - overlap_segment
                    # miss_pred_segment = anchor_length - overlap_segment

                    # 在anchor默认没有重叠的pred的情况下，fn默认等于1
                    # 每当anchor与一个pred产生了重叠，则fn减去当前pred所产生的tp
                    fn_anchor = fn_anchor - overlap_segment / anchor_length

                    tp += overlap_segment / anchor_length
                    # FN += miss_pred_segment / anchor_length
                    fp += min(1.0, error_pred_segment / anchor_length)
                    # if error_pred_segment > anchor_length:
                    #     FP += 1.0
                    # else:
                    #     FP += error_pred_segment / anchor_length
                fn += fn_anchor

            non_overlap_FP = (~ overlap_flag).sum()
            fp += non_overlap_FP
            return tp, fp, fn

    def transfer_prediction(self, output):
        channel_names = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6',
                         'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
                         'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
        prediction = []
        if self.task == 'multi_cls':
            prediction = self.transfer_prediction_multi_cls(output, channel_names)
        if self.task == 'binary_cls':
            prediction = self.transfer_prediction_binary_cls(output, channel_names)

        prediction = prediction.sort_values(by=['start', 'end', '#Channel']).reset_index(drop=True)
        prediction = self.remove_short_prediction(prediction)

        return prediction

    def remove_short_prediction(self, prediction):
        prediction['diff'] = prediction['end'] - prediction['start']
        prediction = prediction[prediction['diff'] >= self.min_l]
        prediction = prediction.drop('diff', axis=1)
        return prediction.reset_index(drop=True)

    def transfer_prediction_multi_cls(self, output, channel_names):
        prediction = []
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                if j == 0:
                    continue
                output_channel_i_label_j = output[:, i, j]
                binary_output = (output_channel_i_label_j > self.th).astype(int)
                start_index, end_index = self.binary_to_index_pairs(binary_output)
                for s, e in zip(start_index, end_index):
                    pred_item = [i, s, e, j]
                    prediction.append(pred_item)
        prediction = pd.DataFrame(prediction, columns=['#Channel', 'start', 'end', 'label'])
        return prediction

    def transfer_prediction_binary_cls(self, output, channel_names):
        prediction = []
        for i in range(output.shape[1]):
            output_channel_i = output[:, i, 0]
            binary_output = (output_channel_i > self.th).astype(int)
            start_index, end_index = self.binary_to_index_pairs(binary_output)
            for s, e in zip(start_index, end_index):
                pred_item = [i, s, e, 0]
                prediction.append(pred_item)
        prediction = pd.DataFrame(prediction, columns=['#Channel', 'start', 'end', 'label'])
        return prediction

    def binary_to_index_pairs(self, binary_output):
        pad_binary_output = np.concatenate([[0], binary_output, [0]])
        start_index = np.where(np.diff(pad_binary_output) == 1)[0]
        end_index = np.where(np.diff(pad_binary_output) == -1)[0]
        return start_index, end_index