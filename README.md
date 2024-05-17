# WaveNet

This is a PyTorch implementation of the WaveNet model for EEG Artifact detection
and classification. The model is based on an invertible wavelet decomposition module
and a U-Net structure. The model is trained on the TUH EEG Artifact dataset
([TUAR](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)).

## Getting Started

### Prerequisites

- Python 3.9
- torch 2.0.1+cu118
- torchaudio 2.0.2+cu118
- torchvision 0.15.2+cu118
- mne 1.4.2
- pywavelets 1.5.0
- networkx 3.2.1
- pandas
- numpy
- matplotlib

### Clone the repository

```bash
git clone https://github.com/miraclefish/WaveNet.git
cd WaveNet
```

### Base Configuration

Change the WORK_SPACE and MONTAGE_PATH in the `base_config.py` file to your own path.

```python
# Current Workspace
WORK_SPACE = '/path/to/your/workspace'

# Montage path for eeg data loading from EDF files.
MONTAGE_PATH = '/path/to/your/montage_path'
```

### Check the test data files

The test data files should be placed in the `data/montage_type/` folder. The test data files should be in the `.edf` format.
Its corresponding annotation file should be in the `.csv` format. 

Due to copyright restrictions on the dataset, we only provide a test sample file `data/01_tcp_ar/aaaaaguk_s002_t001.edf` for running test scripts. 
If you require further in-depth research, please apply to download the complete 
[TUAR](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) dataset from the dataset's official website.

### Check the checkpoint files

The checkpoint files should be placed in the `checkpoints/` folder. The checkpoint files are the model weights saved during the training process.
We provided our best model weights in `WaveNet_78.pth`. You can use this checkpoint file to run the test script.

### Run the test script

```bash
python run_test.py --Resume WaveNet_78 --test_file data/01_tcp_ar/aaaaaguk_s002_t001.edf --device cuda:0
``` 

If the test script runs successfully, you will see the following output:

```bash
**************************************************
Results of test file:  data/01_tcp_ar/aaaaaguk_s002_t001.edf
BASE precision: Pre/eyem: 0.7598        | Pre/chew: 0.0000      | Pre/shiv: 0.0000      | Pre/musc: 0.5112      | Pre/elpp+elec: 0.0000
BASE recall:    Rec/eyem: 0.8270        | Rec/chew: 0.0000      | Rec/shiv: 0.0000      | Rec/musc: 0.5491      | Rec/elpp+elec: 0.0000
BASE f1_score:  F1/eyem: 0.7920 | F1/chew: 0.0000       | F1/shiv: 0.0000       | F1/musc: 0.5295       | F1/elpp+elec: 0.0000
EACS precision: Pre/eyem: 0.4436        | Pre/chew: 0.0000      | Pre/shiv: 0.0000      | Pre/musc: 0.4192      | Pre/elpp+elec: 0.0000
EACS recall:    Rec/eyem: 0.6244        | Rec/chew: 0.0000      | Rec/shiv: 0.0000      | Rec/musc: 0.3125      | Rec/elpp+elec: 0.0000
EACS f1_score:  F1/eyem: 0.5187 | F1/chew: 0.0000       | F1/shiv: 0.0000       | F1/musc: 0.3581       | F1/elpp+elec: 0.0000
clsloss: 0.9108
regloss: 0.0889
loss: 0.9998
**************************************************
```

The reason for the metrics of 0 in the categories CHEW, SHIV, and ELEC is that 
this test file contains very few events of these three types of artifacts. 
Not all types of artifacts are present in every file in the dataset, which is a normal occurrence.

### Run the visualization

The `data_shower.py` is used for EEG data visualization friendly. You can change its parameters to visualize
the EEG data in different ways.

```python

file_path = Path('your/test/file/path')
file_data = FileData(file_path, montageDict, window_size=625) # window_size=625 is not important when visualization
ploter = EEGPloter(file_name=file_path.stem, eeg_data=file_data.eegData,
                   ch_names=file_data.ch_names, label=file_data.label_table)
# time: the start time of the EEG data
# length: the length of the EEG data to be plotted
ploter.plot(time=0, length=10) # plot the first 10 seconds of the EEG data
```


## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details.