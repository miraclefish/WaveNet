import pandas as pd
from visualization import PlotEEGMontage
from dataset_utils import FileData
from dataset_maker.eegLoader import MontageBuilder
from pathlib import Path
from base_config import MONTAGE_PATH


class EEGPloter:

    def __init__(self, file_name, eeg_data, ch_names, label):
        self.file_name = file_name
        self.eeg_data = eeg_data
        self.ch_names = ch_names
        self.label = label
        self.df_data = self.get_dataframe()

    def get_dataframe(self):
        return pd.DataFrame(self.eeg_data.T, columns=self.ch_names)

    def plot(self, **kwargs):
        PlotEEGMontage(eeg_signal=self.df_data, file_name=self.file_name, label=self.label, **kwargs)


if __name__ == '__main__':
    montagePath = Path(MONTAGE_PATH)

    montageDict = MontageBuilder(path=montagePath).initial_montage()
    file_path = Path('/root/autodl-pub/YYF/WaveNet/data/01_tcp_ar/aaaaaguk_s002_t001.edf')
    file_data = FileData(file_path, montageDict, window_size=625)
    ploter = EEGPloter(file_name=file_path.stem, eeg_data=file_data.eegData,
                       ch_names=file_data.ch_names, label=file_data.label_table)
    ploter.plot(time=0, length=10)
    pass
