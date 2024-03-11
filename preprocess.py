# %%
import logging
import shutil
import os
from pathlib import Path
import scipy as sp

import more_itertools as mit
import numpy as np
import pandas as pd
import hydra
from omegaconf import OmegaConf, DictConfig

import tools as T

log = logging.getLogger(__name__)

# %%
def generate_signal(label, mean_time, sr):
    """
    Generate a signal for a given label

    Parameters
    ----------
    label : str
        label of the signal

    Returns
    -------
    signal : np.ndarray
        random signal
    """
    t_signal = np.random.normal(mean_time, 0.1*mean_time)
    x = np.arange(t_signal*sr)
    timestamp = x / sr
    
    if label == 'M':
        # sinusoid with amplitude 1, period 0.01s
        signal = np.sin(2 * np.pi / 0.01 * timestamp + np.random.rand() ) + np.random.normal(0, 0.3, len(timestamp))
    elif label == 'N':
        # sinusoid with amplitude 2, period 0.02s
        signal = 2*np.sin(2 * np.pi / 0.02 * timestamp + np.random.rand()) + np.random.normal(0, 0.5, len(timestamp))
    elif label == 'NL':
        # sinusoid with amplitude 5, period 0.015s
        signal = 5*np.sin(2 * np.pi / 0.015 * timestamp + np.random.rand()) + np.random.normal(0, 1.0, len(timestamp))
    else:
        raise ValueError(f'Unknown label: {label}')
    return signal

def generate_toy(cfg):
    """
    # Generate toy data as we cannot release original patient data

    Parameters
    ----------
    cfg : omegaconf.DictConfig object

    Returns
    -------
    cohort : pd.DataFrame
        columns are [label, muscle_type, filepath]
    """
    # Summary of the original data
    summary = {
        'n_patients': [19, 19, 19],
        'n_signals': [122, 160, 94],
        'n_proximal': [64, 63, 17],
        'n_distal': [58, 97, 77],
        'total_time': [312.84, 422.78, 203.50],
        'label': ['M', 'N', 'NL'],
    }
    summary = pd.DataFrame(summary)
    summary.set_index(pd.Index(['M', 'N', 'NL']), inplace=True)
    
    # Generate patient_id
    patient_id = np.arange(1, sum(summary['n_patients'])+1)
    patient_id_dict = {label: patient_id[i1:i2] for label, (i1, i2) in zip(summary.index, mit.pairwise(np.cumsum([0]+summary['n_patients'].values.tolist())))}
        
    cohort = {'label': [], 'muscle_type': [], 'filepath': []}

    for i, (n_patient, n_signals, n_proximal, n_distal, total_time, label) in summary.iterrows():
        
        # Randomly assign n_signals per patient
        n_patient = int(n_patient)
        n_signal_per_patient = np.diff(np.sort(np.append(np.random.choice(n_signals, n_patient-1, replace=False), [0, n_signals])))

        # Assign proximal/distal signals per patient
        n_signal_per_patient_ = n_signal_per_patient.copy()
        n_proximal_per_patient = np.zeros(n_patient, dtype=int)
        for i in range(n_proximal):
            i_patient = np.random.choice(np.where(n_signal_per_patient_>0)[0])
            n_proximal_per_patient[i_patient] += 1
            n_signal_per_patient_[i_patient] -= 1
        n_distal_per_patient = n_signal_per_patient - n_proximal_per_patient
        
        # Generate signals, append to dataframe (cohort), save file
        mean_time = total_time / n_signals
        
        shutil.rmtree(f'{cfg.dir.data_raw}/{label}', ignore_errors=True)
        os.makedirs(f'{cfg.dir.data_raw}/{label}', exist_ok=True)
        for i, (i_patient, n_signals) in enumerate(zip(patient_id_dict[label], n_signal_per_patient)):
            # Save as ${label}/${i_patient}_{i_signal}.p at ${cfg.dir.data_raw}
            for i_signal in range(n_signals):
                signal = generate_signal(label, mean_time, cfg.sr_src)
                T.save_pickle(signal, f'{cfg.dir.data_raw}/{label}/{i_patient}_{i_signal}.p')

                muscle_type = 'P' if i_signal < n_proximal_per_patient[i] else 'D'
                cohort['label'].append(label)
                cohort['muscle_type'].append(muscle_type)
                cohort['filepath'].append(f'{cfg.dir.data_raw}/{label}/{i_patient}_{i_signal}.p')

    cohort = pd.DataFrame(cohort)
    cohort.to_csv(f'{cfg.dir.data_raw}/cohort.csv', encoding='utf-8', index=False)
    return cohort

def segmentize(signal, nperseg, noverlap):
    # Discards last segment if it is smaller than nperseg
    n = np.arange(0, len(signal)-nperseg, noverlap)
    segments = np.lib.stride_tricks.as_strided(signal, shape=(len(n), nperseg), strides=(signal.strides[0]*noverlap, signal.strides[0]))
    return segments

def segmentize_and_save(signal, nperseg, noverlap, row, path_save):
    """
    Slice a signal into segments and save them

    Parameters
    ----------
    signal : array-like of shape (n_samples,), default=None
        Argument explanation.
        If ``None`` is given, those that appear at least once
        .. versionadded:: 0.18
    Returns
    -------
    df_signal : dataframe with columns [filepath, label, muscle]
    """
    filename = os.path.basename(row['filepath']).split('.')[0] #'data/raw/${label}/123456_0.p' -> '123456_0'
    label = row['label']
    muscle_type = row['muscle_type']
    
    # Slice 
    segments = segmentize(signal, nperseg, noverlap)

    info = []
    for i, segment in enumerate(segments):
        filepath = path_save/(filename+f'_{i}.p')
        T.save_pickle(segment, filepath)
        info.append([filepath, label, muscle_type])
    df_signal = pd.DataFrame(info, columns = ['filepath', 'label', 'muscle_type'])

    return df_signal

# %%
@hydra.main(config_path='conf', config_name='preprocess', version_base='1.2')
def main(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    # Load cohort information (label, muscle_type, filepath)
    if cfg.toy:
        cohort = generate_toy(cfg)
    else:
        cohort = pd.read_csv(cfg) # Bring your own cohort.csv
    labels = sorted(cohort['label'].unique())

    # Reset preprocessed data directory
    path_data = Path(cfg.dir.data_processed)
    shutil.rmtree(path_data, ignore_errors=True)
    for label in labels:
        (path_data/label).mkdir(parents=True)
    
    # Preprocess (downsample, segmentize, save segments)
    nperseg = int(cfg.window * cfg.sr_trg) # length of a segment in data points
    noverlap = int((cfg.window - cfg.overlap) * cfg.sr_trg) # length of a noverlap in data points

    df_signal_list = []
    log.info('Slicing signal into windowed dsignals...')
    for i, row in cohort.iterrows():
        path_save = Path(path_data/row['label'])
        print_content = str(row['filepath'])+' '

        # Load signal
        signal = T.load_pickle(row['filepath'])

        # Resample (Downsample)
        T_signal = len(signal)
        s = T_signal/cfg.sr_src
        T_hat = int(s * cfg.sr_trg)
        signal = sp.signal.resample(signal, T_hat)
        print_content += f'resampled... {T_signal}->{T_hat}'
        
        df_signal = segmentize_and_save(signal, nperseg, noverlap, row, path_save)
        df_signal_list.append(df_signal)
        log.info(print_content)
    log.info('done')
    cohort_new = pd.concat(df_signal_list, axis=0, ignore_index=True)
    
    # %%
    # Summarize into log
    content = str(cohort_new['label'].value_counts())+'\n'
    content += str(cohort_new['muscle_type'].value_counts())+'\n'
    log.info(content)
    T.write(content, path_save/'summary.txt')

    log.info(f'Number of files: [{len(cohort_new)}]')

    # %%
    # Save segment cohort file
    cohort_new.to_csv(path_data/'cohort_signal.csv', encoding='utf-8', index=False)

if __name__=='__main__':
    main()


