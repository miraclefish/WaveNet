import numpy as np
import torch
from typing import Iterable
from dataset_utils import FileData


def predict_for_test(
        model: torch.nn.Module,
        data_loader: Iterable,
        dataset: FileData,
        loss_fn: torch.nn.Module,
        device: torch.device,
):
    model.eval()

    loss_reg = 0
    loss_cls = 0
    loss_total = 0

    base_label = []
    prediction = []

    n_batch = len(data_loader)
    for i, (batch) in enumerate(data_loader):
        batch_x, batch_label = batch['data'], batch['label']

        input_x = batch_x.float().to(device)
        label = batch_label.float().to(device)

        with torch.no_grad():
            output = model(input_x)
            batch_loss_reg = output['loss_reg']
            batch_loss_cls = loss_fn(output['y_pred'], label)

            batch_loss = batch_loss_cls + batch_loss_reg

            loss_reg += batch_loss_reg.item()
            loss_cls += batch_loss_cls.item()
            loss_total += batch_loss.item()

        preparation_for_subject_analysis(
            output=output,
            label=label,
            prediction=prediction,
            base_label=base_label,
        )


    prediction = np.concatenate(prediction, axis=0)
    prediction = prediction.reshape(-1, *prediction.shape[2:])
    base_label = np.concatenate(base_label, axis=0)
    base_label = base_label.reshape(-1, *base_label.shape[2:])

    loss_reg = loss_reg / n_batch
    loss_cls = loss_cls / n_batch
    loss_total = loss_total / n_batch


    test_state = {
        'regloss': loss_reg,
        'clsloss': loss_cls,
        'loss': loss_total,
        'prediction': {0: prediction},
        'base_label': {0: base_label},
        'event_label': {0: dataset.label_table},
    }

    return test_state


def preparation_for_subject_analysis(
        output: torch.Tensor,
        label: torch.Tensor,
        prediction,
        base_label,
):
    pred = np.transpose(output['y_pred'].cpu().numpy(), [0, 2, 1, 3])
    b_l = np.transpose(label.cpu().numpy(), [0, 2, 1, 3])

    prediction.append(pred)
    base_label.append(b_l)
    return None

