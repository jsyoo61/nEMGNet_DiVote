import itertools as it
import logging
import os
import pprint

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig, open_dict
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
import torch.utils.data as D

# %%
import sys
sys.path.append('/home/jaesungyoo/programming/hideandseek')
# PROJECT_DIR='/home/jaesungyoo/EMG'
# os.chdir('/home/jaesungyoo/EMG/train')

# %%
import dataset as DAT
import eval as E
import tools as T
import utils as U

# %%
log = logging.getLogger(__name__)
#

# %% codecell
if False:
    # %%
    p = T.Path(PROJECT_DIR)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=p.join('conf'), job_name='debug')
    overrides = [
    # f"EXP_DIR=exp/train1/proposed/0",
    # f"path.train1_dir=exp/train1/proposed/1",
    f"path.train1_dir=exp/train1/NAS3/409",
    ]
    cfg = hydra.compose(config_name='train2_config', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))

    # %%
    os.getcwd()
    log = T.TDict()
    log.info = print
    log.debug = print

    # %%
    # Dummy class for debugging
    class Dummy():
        """ Dummy class for debugging """
        def __init__(self):
            pass
    self = Dummy()
    ex_type='all'
    # os.makedirs(p.join('train/outputs/train2/dummy'), exist_ok=True)
    # os.chdir(p.join('train/outputs/train2/dummy'))
    # %%
    # model = model1
    # loader = train_loader
    # cohort = train_cohort
    #
    # gb=it.groupby(cohort[['person_id', 'NMNL', 'PD']].values, key=lambda x: x[0])
    # person_id, g = next(gb)
    # person_id
    # l

# %%
class SubFeatExtracter:
    allowed_ex_types = ['all', 'PD']
    def __init__(self, ex_type, model):

        if ex_type not in self.allowed_ex_types:
            raise Exception(f'ex_type (extraction type) must be one of {self.allowed_ex_types}, received: {ex_type}')
        self.ex_type = ex_type

        self.model = model

    def extract(self, loader, cohort):
        log.info('[SubFeatExtracter.extract] extracting subject features...')
        device = T.torch.get_device(self.model)

        y_list = []
        y_score_list = []
        with torch.no_grad():
            for data in loader:
                x = data['x'].to(device)
                y = data['y'].numpy()
                y_hat = self.model(x)
                y_score = torch.softmax(y_hat, dim=1).cpu().numpy()

                y_list.append(y)
                y_score_list.append(y_score)
        y = np.concatenate(y_list, axis=0)
        y_score = np.concatenate(y_score_list, axis=0)

        y_encoder = loader.dataset.y_encoder.classes_.tolist()

        # Soft vote within muscle signal
        i=0
        results_wf = {'wavefile_name':[], 'y':[], 'y_label':[], 'y_score':[]}
        for wavefile_name, g in it.groupby(cohort[['wavefile_name', 'NMNL', 'PD']].values, key=lambda x: x[0]):
            l = np.array(list(g))
            n_frames = len(l)
            y_true= l[0,1]

            y_ = y[i:i+n_frames]
            y_score_ = y_score[i:i+n_frames]

            try:
                assert T.numpy.equal(y_)
                assert T.numpy.equal(l[:,:2])
            except:
                import pdb; pdb.set_trace()

            y_ = int(y_[0])
            assert y_true==y_encoder[y_], f'y_true: ({y_true}), y_encoder: ({y_encoder[y_]}), i: ({i})'

            results_wf['wavefile_name'].append(wavefile_name)
            results_wf['y'].append(y_)
            results_wf['y_label'].append(y_true)
            results_wf['y_score'].append(y_score_.mean(axis=0))

            i += n_frames

        for k,v in results_wf.items():
            results_wf[k] = np.array(v)

        # Soft vote within all/PD muscle to create subject features
        i=0
        results_sf = {'person_id':[], 'y':[], 'y_label':[], 'y_score':[]}
        for person_id, g in it.groupby(cohort[['person_id', 'NMNL', 'PD']].values, key=lambda x: x[0]):
            l = np.array(list(g))
            n_frames = len(l)
            y_true= l[0,1]

            y_ = y[i:i+n_frames]
            y_score_ = y_score[i:i+n_frames]
            try:
                assert T.numpy.equal(y_)
                assert T.numpy.equal(l[:,:2])
            except:
                import pdb; pdb.set_trace()
            y_ = int(y_[0])
            assert y_true==y_encoder[y_], f'y_true: ({y_true}), y_encoder: ({y_encoder[y_]}), i: ({i})'

            results_sf['person_id'].append(person_id)
            results_sf['y'].append(y_)
            results_sf['y_label'].append(y_true)
            if self.ex_type == 'all':
                results_sf['y_score'].append(y_score_.mean(axis=0))
            elif self.ex_type == 'PD':
                P_i = l[:,2]=='P'
                D_i = l[:,2]=='D'
                assert (~P_i == D_i).all()
                P_score = y_score_[P_i].mean(axis=0)
                D_score = y_score_[D_i].mean(axis=0)
                feature = np.concatenate((P_score,D_score), axis=0)
                feature = np.nan_to_num(feature, nan=1/3) # Impute nan (missing) with [0.333,0.333,0.333]
                results_sf['y_score'].append(feature)
            else:
                raise Exception(f'ex_type (extraction type) must be one of ["all", "PD"], received: {self.ex_type}')

            i += n_frames

        for k,v in results_sf.items():
            results_sf[k] = np.array(v)

        if self.ex_type=='all':
            # Plot featuremap
            pass

        extracted_data = {
        'wavefiles': results_wf,
        'subject_features': results_sf,
            # {
            # 'x': results_sf['y_score'],
            # 'y': results_sf['y'],
            # }
        }
        return extracted_data

def get_dataset(wav_dir, cohort, exp_cfg, train):
    if exp_cfg.torch_dataset._target_=='dataset.SimpleEMGDataset':
        dataset = hydra.utils.instantiate(exp_cfg.torch_dataset, wav_dir = wav_dir, cohort=cohort,
                                x_channel=exp_cfg.x_channel, y_array=exp_cfg.y_array)
    elif exp_cfg.torch_dataset._target_=='dataset.MelDataset':
        if train:
            dataset = hydra.utils.instantiate(exp_cfg.torch_dataset, wav_dir=wav_dir, cohort=cohort, sr=exp_cfg.method.sr_trg, scale=True, hop_length=exp_cfg.hop_length)
        else:
            dataset = hydra.utils.instantiate(exp_cfg.torch_dataset, wav_dir=wav_dir, cohort=cohort, sr=exp_cfg.method.sr_trg, scale=False, hop_length=exp_cfg.hop_length)
    elif exp_cfg.torch_dataset._target_=='dataset.SignalToImageDataset':
        dataset = hydra.utils.instantiate(exp_cfg.torch_dataset, wav_dir=wav_dir, cohort=cohort)
    else:
        raise Exception(f'Unknown dataset: {exp_cfg.torch_dataset}')
    return dataset

# %%
@hydra.main(config_path='../conf', config_name='train2_config')
def main(cfg: DictConfig) -> None:
    import warnings
    warnings.filterwarnings('ignore')
# %%
    # Print current experiment info
    device, path = U.exp_setting(cfg)

    # %%
    # Experiment1 paths
    exp1_cfg = OmegaConf.load(cfg.path.train1_cfg_dir)

    # Add Experiment1 config to cfg and Save.
    cfg_to_save=OmegaConf.to_container(cfg)
    cfg_to_save['exp1_cfg'] = OmegaConf.to_container(exp1_cfg)
    cfg_to_save = OmegaConf.create(cfg_to_save)
    # %%
    OmegaConf.save(cfg_to_save, '.hydra/config.yaml')

    # %%
    # Transfer necessary configs from cfg1
    with open_dict(cfg):
        cfg.encoding = exp1_cfg.encoding
        cfg.method = exp1_cfg.method

    exp1_path = T.Path(cfg.path._train1_dir)
    exp1_path.model = 'model'
    exp1_path.node = 'node'

    # %%
    # Create data from model1's predictions
    # Get exp1's dataset, Ignore validation, will not use here.
    cohort = pd.read_csv(cfg.path.cohort)
    train_i = T.load_pickle(exp1_path.join('train_i.p'))
    test_i = T.load_pickle(exp1_path.join('test_i.p'))
    train_cohort = E.refine_cohort(cohort.iloc[train_i], exp1_cfg)
    test_cohort = E.refine_cohort(cohort.iloc[test_i], exp1_cfg)

    # %%
    # Load Data
    '''May use RestingEMGDataset for training,
    and SimpleEMGDataset for validation/testing'''
    dataset_train = hydra.utils.instantiate(exp1_cfg.torch_dataset, cohort=train_cohort)
    dataset_test = hydra.utils.instantiate(exp1_cfg.torch_dataset, cohort=test_cohort)
    if hasattr(dataset_test, 'scaler'):
        DAT.transfer_scalers(dataset_train, dataset_test)
    assert DAT.consistency_check(dataset_train, dataset_test)

    train_loader = D.DataLoader(dataset_train, batch_size=cfg.test_batch_size, shuffle=False)
    test_loader = D.DataLoader(dataset_test, batch_size=cfg.test_batch_size, shuffle=False)

    y_encoder = dataset_train.y_encoder.classes_.tolist()

    # %%
    # Loading 1st model
    exp1_cfg.model = E.version_check(exp1_cfg.model) # Clean up the mess
    model1 = hydra.utils.instantiate(exp1_cfg.model, n_classes = len(dataset_train.y_encoder.classes_))
    model1_path = exp1_path.node.join('model.pt')
    log.info(f'loading from: [{model1_path}]')
    state_dict = E.load_state_dict(model1_path) # Clean up the mess
    model1.load_state_dict(state_dict)
    model1.to(device)
    model1.eval()

    # %%
    subfeat_extracter = hydra.utils.instantiate(cfg.subfeatextract, model=model1)

    subfeat_train = subfeat_extracter.extract(train_loader, train_cohort)
    subfeat_test = subfeat_extracter.extract(test_loader, test_cohort)

    data_train = {
    'x': subfeat_train['subject_features']['y_score'],
    'y': subfeat_train['subject_features']['y']
    }
    data_test = {
    'x': subfeat_test['subject_features']['y_score'],
    'y': subfeat_test['subject_features']['y']
    }

    # %%
    # def generate_dummy(n=10):
    #     l_x, l_y = [], []
    #     for i in range(3):
    #         x = np.random.rand(n,3)
    #         x[:,i] *= 3
    #         x = x/x.sum(1)[...,None]
    #         l_x.append(x)
    #
    #         y = np.full(n, i)
    #         l_y.append(y)
    #     x, y = np.concatenate(l_x), np.concatenate(l_y)
    #     return x, y
    #
    # # %%
    # x, y = generate_dummy(100)
    # data_train = {
    # 'x': x,
    # 'y': y
    # }
    #
    # x, y = generate_dummy(10)
    # data_test = {
    # 'x': x,
    # 'y': y
    # }
    # import hideandseek as hs
    # hs.E.classification_report_full(result_train)
    # import sklearn.metrics as metrics
    # metrics.classification_report(result_train['y_true'], result_train['y_pred'], output_dict=True)

    # %%
    data_subfeat = {
    'train': data_train,
    'test': data_test,
    }

    # %%
    model1.cpu()
    del model1
    torch.cuda.empty_cache()

    # %%
    model2 = LogisticRegression()
    # model2 = LogisticRegression(multi_class='ovr')
    model2.fit(data_subfeat['train']['x'], data_subfeat['train']['y'])
    T.save_pickle(model2, path.model.join('model2.p'))
    log.info('Saved model2')

    # %%
    y_score_train = model2.predict_proba(data_subfeat['train']['x'])

    result_train = {
    'x': data_subfeat['train']['x'],
    'y_true': data_subfeat['train']['y'],
    'y_score': y_score_train,
    'y_pred': y_score_train.argmax(1)
    }

    y_score_test = model2.predict_proba(data_subfeat['test']['x'])

    result_test = {
    'x': data_subfeat['test']['x'],
    'y_true': data_subfeat['test']['y'],
    'y_score': y_score_test,
    'y_pred': y_score_test.argmax(1)
    }

    # %%
    # Evaluations
    scorer = E.Scorer(RESULT_DIR='result')

    # %%
    # Testing: Best model
    scores = scorer.score_subject2(model_skl=model2, result_train=result_train, result_test=result_test, y_encoder=y_encoder, cfg=cfg)
    scores = T.unnest_dict(scores)

    # %%
    pp=pprint.PrettyPrinter(indent=2)
    content = pp.pformat(scores)
    log.info(content)
    T.write(content, path.result.join('scores.txt'))
    T.save_pickle(scores, 'scores.p')

# %%
if __name__ == '__main__':
    main()
