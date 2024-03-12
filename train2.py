# %%
import itertools as it
import logging
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch

# %%
import dataset as D
import tools as T
import hideandseek as hs

# %%
log = logging.getLogger(__name__)
#

# %% codecell
if False:
    # %%
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(os.getcwd(),'conf'), job_name='train')
    overrides = ['dir.train1=exp/train1']
    cfg = hydra.compose(config_name='train2', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    log.info = print

# %%
def extract_features(dataset, cohort, network, feature_type, test_batch_size=64):
    """
    Extract patient features given a trained network and segment dataset.
        Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Segment dataset to extract
    cohort : pd.DataFrame
        Cohort dataframe including filepath, muscle_type
        patient_id, signal_id is inferred from filepath (e.g. 'P0001_01_01.p' -> 'P0001')
    network : torch.nn.Module
        Trained network to extract features

    feature_type : str ['all', 'muscle']
        Type of patient feature to extract.
        'all' soft votes signal_y_score across all signals
        'muscle' soft votes signal_y_score across all signals within muscle_type, and concatenates into a longer vector
          
    Returns
    -------
    data : dict
        Dictionary containing patient features ('x'), patient labels ('y_true'), and patient dataframe ('df', columns=['patient_id', 'y_true', 'x'])
    """

    assert feature_type in ['all', 'muscle'], f'feature_type must be one of ["all", "muscle"], received: {feature_type}'

    # Extract segment prediction scores
    segment_result_dict = hs.E.test_network(network, dataset, batch_size=test_batch_size)

    cohort = cohort.reset_index(drop=True)
    cohort['patient_id'] = cohort.apply(lambda x: os.path.basename(x['filepath']).split('_')[0], axis=1)
    cohort['signal_id'] = cohort.apply(lambda x: os.path.basename(x['filepath']).split('_')[1], axis=1)
    cohort['y_true'] = segment_result_dict['y_true']

    # Soft vote segment prediction scores to make signal prediction scores
    signal_pred_results = []
    for (patient_id, signal_id), df in cohort.groupby(by=['patient_id', 'signal_id']):
        assert T.equal(df['y_true']) and T.equal(df['muscle_type'])
        y_true, muscle_type, signal_y_score = df['y_true'].iloc[0], df['muscle_type'].iloc[0], segment_result_dict['y_score'][df.index].mean(axis=0)
        signal_pred_results.append([patient_id, signal_id, y_true, muscle_type, signal_y_score])
    df_signal_pred = pd.DataFrame(signal_pred_results, columns=['patient_id', 'signal_id', 'y_true', 'muscle_type', 'y_score'])
    signal_y_score = np.stack(df_signal_pred['y_score'].to_numpy(), axis=0)

    # Soft vote signal prediction scores to make patient features
    patient_features = []
    if feature_type == 'all':
        for patient_id, df in df_signal_pred.groupby(by=['patient_id']):
            assert T.equal(df['y_true'])
            patient_features.append([patient_id, df['y_true'].iloc[0], signal_y_score[df.index].mean(axis=0)])
    elif feature_type == 'muscle':
        muscle_type_list = sorted(['P', 'D'])
        n_class = signal_y_score.shape[1]

        for (patient_id, muscle_type), df in df_signal_pred.groupby(by=['patient_id', 'muscle_type']):
            assert T.equal(df['y_true'])
            patient_features.append([patient_id, muscle_type, df['y_true'].iloc[0], signal_y_score[df.index].mean(axis=0)])
        df_muscle_type_pred = pd.DataFrame(patient_features, columns=['patient_id', 'muscle_type', 'y_true', 'y_score'])
        
        # Concatenate Proximal & Distal muscle scores to create patient features
        patient_features = []
        for patient_id, df in df_muscle_type_pred.groupby(by=['patient_id']):
            assert T.equal(df['y_true'])

            existing_muscle_types = df['muscle_type'].unique()
            x_muscle_type = [df[df['muscle_type']==muscle_type]['y_score'].iloc[0] if muscle_type in existing_muscle_types
                             else np.full((n_class), 1/n_class)
                             for muscle_type in muscle_type_list]
            x = np.concatenate(x_muscle_type, axis=0)

            patient_features.append([patient_id, df['y_true'].iloc[0], x])

    df_patient_features = pd.DataFrame(patient_features, columns=['patient_id', 'y_true', 'x'])
    patient_features = np.stack(df_patient_features['x'].to_numpy(), axis=0)

    data = {
        'df': df_patient_features,
        'x': patient_features,
        'y_true': df_patient_features['y_true'].to_numpy(),
    }

    return data

# %%
@hydra.main(config_path='conf', config_name='train2', version_base='1.2')
def main(cfg: DictConfig) -> None:
    # %%
    # Print current experiment info
    log.info(OmegaConf.to_yaml(cfg))

    # Set GPU for current experiment if there's multiple gpu in the environment
    device = T.torch.multiprocessing_device(gpu_id=cfg.gpu_id)
    T.torch.seed(cfg.random.seed, strict=cfg.random.strict)
    log.info(f'device: {device}')

    # %%
    # Assumes the process runs in a new directory (hydra.job.cwd==True)
    path_dict = {'train1': Path(cfg.dir._train1), 'classifier_dir': Path('classifier')}
    path_dict['network'] = path_dict['train1']/'network/network.pt'
    path_dict['classifier'] = path_dict['classifier_dir']/'classifier.p'
    path_dict['classifier_dir'].mkdir(parents=True, exist_ok=True)

    # %%
    # Load train1 cfg
    cfg_train1 = OmegaConf.load(cfg.dir.train1_cfg)

    # Add train1 cfg to current cfg and Save.
    cfg_to_save=OmegaConf.to_container(cfg)
    cfg_to_save['train1'] = OmegaConf.to_container(cfg_train1)
    cfg_to_save = OmegaConf.create(cfg_to_save)
    OmegaConf.save(cfg_to_save, '.hydra/config.yaml')

    # %%
    # Neural network
    network = hydra.utils.instantiate(cfg_train1.nn)
    network.load_state_dict(torch.load(path_dict['network']))
    network.to(device)
    log.info(f'Loading network from: [{path_dict["network"]}]')

    # %%
    # Load data (Validation not used)
    ds_train, ds_val, ds_test = D.get_dataset(cfg_train1)

    cohort = pd.read_csv(cfg_train1.dir.data_cohort)
    train_i, test_i = T.load_pickle(path_dict['train1']/'train_i.p'), T.load_pickle(path_dict['train1']/'test_i.p')
    cohort_train, cohort_test = cohort.iloc[train_i], cohort.iloc[test_i]

    # Sanity check on train/val/test split
    assert cohort_train['filepath'].tolist() == ds_train.x_i and cohort_test['filepath'].tolist() == ds_test.x_i
    
    # %%
    # Extract patient features (data is patient level not signal level)
    data_patient_train = extract_features(ds_train, cohort_train, network, feature_type=cfg.feature_type, test_batch_size=cfg.test_batch_size)
    data_patient_test = extract_features(ds_test, cohort_test, network, feature_type=cfg.feature_type, test_batch_size=cfg.test_batch_size)
    network.cpu()
    del network
    torch.cuda.empty_cache()

    # %%
    # Simple averaging without classifier just to compare performance
    if cfg.feature_type=='all':
        result_no_classifier = {
            'y_true': data_patient_test['y_true'],
            'y_score': data_patient_test['x'],
            'y_pred': data_patient_test['x'].argmax(1)
        }
        scores = hs.E.evaluate(results=result_no_classifier, metrics=hs.E.classification_report_full)
        log.info(f'[No classifier] scores: {pd.DataFrame(scores).T}')

    # %%
    # Add classifier to patient features
    classifier = LogisticRegression()
    classifier.fit(data_patient_train['x'], data_patient_train['y_true'])
    T.save_pickle(classifier, path_dict['classifier'])
    log.info(f'Saved classifier at {path_dict["classifier"]}')
    
    # %%
    y_score = classifier.predict_proba(data_patient_test['x'])
    result = {
        'y_true': data_patient_test['y_true'],
        'y_score': y_score,
        'y_pred': y_score.argmax(1)
    }
        
    scores = hs.E.evaluate(results=result, metrics=hs.E.classification_report_full)
    log.info(f'Scores: {pd.DataFrame(scores).T}')
    T.save_pickle(scores, 'scores.p')

    if cfg.save_result:
        T.save_pickle(result, 'result.p')

# %%
if __name__ == '__main__':
    main()
