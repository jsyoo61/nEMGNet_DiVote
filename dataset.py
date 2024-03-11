'''dataset class which loads data and returns it'''
import logging
import os

import hydra
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

import tools as T
import hideandseek as hs

# %%
log = logging.getLogger(__name__)

# %%
def get_dataset(cfg):
    # Load signal cohort
    cohort = pd.read_csv(cfg.dir.data_cohort)
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(cohort['label'])
    labels = y_encoder.classes_
    cohort['y'] = y

    # Get patient cohort from signal cohort (Could directly load patient cohort)
    patient_label = {}
    patient_signal_list = {}
    cohort['patient_id'] = cohort.apply(lambda x: os.path.basename(x['filepath']).split('_')[0], axis=1)
    for i, (patient_id, y) in cohort[['patient_id', 'y']].iterrows():
        if patient_id in patient_label:
            assert patient_label[patient_id] == y, f"Inconsistent y for the same patient_id: {patient_id}: {patient_label[patient_id]} vs {y}"
            patient_signal_list[patient_id].append(i)
        else:
            patient_label[patient_id] = y
            patient_signal_list[patient_id] = [i]

    cohort_patient = pd.DataFrame(patient_label.items(), columns=['patient_id', 'y'])
    
    # Split patients
    train_patient_i, val_patient_i, test_patient_i = hydra.utils.instantiate(cfg.cv, y=cohort_patient['y'].values) # get indices based on y value (Could use Stratified split)
    patient_id = cohort_patient['patient_id'].to_numpy()
    train_patient, val_patient, test_patient = patient_id[train_patient_i], patient_id[val_patient_i], patient_id[test_patient_i]

    # Split signals (Not the exact desired train/val/test ratio)
    def get_signal_i(patients):
        return np.concatenate([patient_signal_list[patient] for patient in patients])
    train_i, val_i, test_i = get_signal_i(train_patient), get_signal_i(val_patient), get_signal_i(test_patient)
    log.info(f'train/val/test: {len(train_i)/len(cohort)}/{len(val_i)/len(cohort)}/{len(test_i)/len(cohort)}')    
    train_cohort, val_cohort, test_cohort = cohort.iloc[train_i], cohort.iloc[val_i], cohort.iloc[test_i]

    # Make torch dataset
    ds_train, ds_val, ds_test = nEMGDataset(train_cohort), nEMGDataset(val_cohort), nEMGDataset(test_cohort)

    return ds_train, ds_val, ds_test

# %%
class nEMGDataset(hs.D.Dataset):
    targets_type = 'categorical'
    def __init__(self, cohort):
        self.x_i = cohort['filepath'].tolist()
        self.y = torch.as_tensor(cohort['y'].to_numpy(), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def get_x(self, idx):
        signal = T.load_pickle(self.x_i[idx])
        signal = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0) # Add channel dimension
        return signal 

    def get_y(self, idx):
        return self.y[idx]
    
    def get_y_all(self):
        return self.y
    
# %%
