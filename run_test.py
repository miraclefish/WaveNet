from pathlib import Path
import argparse
import torch
from torch.utils.data import DataLoader

from dataset_utils.dataset import FileData
from utils import get_model, create_loss_fn, seed_everything
from dataset_maker.eegLoader import MontageBuilder
from experiment import ModelCheckpoint, predict_for_test, BASEevaluator, EACSevaluator
from base_config import WORK_SPACE, MONTAGE_PATH


def get_args():
    parser = argparse.ArgumentParser('WaveNet test script', add_help=False)

    # running parameters
    parser.add_argument('--Resume', default='WaveNet_78', type=str)
    parser.add_argument('--ckpt_folder', default='./checkpoints', type=str)
    parser.add_argument('--test_file', default='data/01_tcp_ar/aaaaaguk_s002_t001.edf', type=str)

    # device parameters
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    return args


def main(args):

    device = torch.device(args.device)
    print('device: ', device)

    ckpt_folder = Path(WORK_SPACE) / args.ckpt_folder

    if args.Resume is None:
        print("You need to provide a Resume version model ckpt for test.")
        exit(1)

    else:
        file_name = f'{args.Resume}.pth'
        ckpt = ModelCheckpoint(path=ckpt_folder, filename=file_name)

        args = ckpt.load_args_from_ckpt(args)
        seed = seed_everything(args.seed)
        model = get_model(args)

        model.to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print("Model = %s" % str(model))
        print('number of params:', n_parameters)
        print('args:', args)

        epochs = ckpt.load(model, device)

    montagePath = Path(MONTAGE_PATH)
    montageDict = MontageBuilder(path=montagePath).initial_montage()
    test_set_path = Path(WORK_SPACE) / args.test_file
    print(test_set_path)
    test_set = FileData(file_path=test_set_path, montageDict=montageDict, window_size=args.window_size, overlap=0.,
                        clip=args.clip, norm_method=args.norm_method, task=args.task)
    dataloader_test = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_works)

    loss_fn = create_loss_fn(args)

    predict_state = predict_for_test(
        model=model,
        dataset=test_set,
        data_loader=dataloader_test,
        loss_fn=loss_fn,
        device=device,
    )

    EvalBASE = BASEevaluator(
        prediction=predict_state['prediction'],
        base_label=predict_state['base_label'],
        task=args.task,
    )

    base_precision, base_recall, base_f1_score = EvalBASE.get_metrics()

    EvalEACS = EACSevaluator(
        prediction=predict_state['prediction'],
        event_label=predict_state['event_label'],
        task=args.task,
    )

    eacs_precision, eacs_recall, eacs_f1_score = EvalEACS.get_metrics()

    print('*' * 50)
    print('Results of test file: ', args.test_file)
    print('BASE precision: ' + '\t| '.join([f'{key}: {val:.4f}' for key, val in base_precision.items()]))
    print('BASE recall: \t' + '\t| '.join([f'{key}: {val:.4f}' for key, val in base_recall.items()]))
    print('BASE f1_score: \t' + '\t| '.join([f'{key}: {val:.4f}' for key, val in base_f1_score.items()]))
    print('EACS precision: ' + '\t| '.join([f'{key}: {val:.4f}' for key, val in eacs_precision.items()]))
    print('EACS recall: \t' + '\t| '.join([f'{key}: {val:.4f}' for key, val in eacs_recall.items()]))
    print('EACS f1_score: \t' + '\t| '.join([f'{key}: {val:.4f}' for key, val in eacs_f1_score.items()]))

    print('clsloss: {:.4f}'.format(predict_state['clsloss']))
    print('regloss: {:.4f}'.format(predict_state['regloss']))
    print('loss: {:.4f}'.format(predict_state['loss']))
    print('*' * 50)

    pass


if __name__ == '__main__':
    args = get_args()
    main(args)
