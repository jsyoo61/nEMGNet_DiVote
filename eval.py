'''Contain testing functions.
When run, runs multiple tests of each experiment in specified multirun folder '''
import collections
from copy import deepcopy as dcopy
import itertools as it
from itertools import combinations, groupby
import logging
import multiprocessing
import os
import pathlib
import pprint

import hydra
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from scipy import signal
import seaborn as sns
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.utils.data as D

# %%
import hideandseek as hs
import tools as T
import tools.numpy
import tools.torch
import tools.plot
import utils as U
# from utils import list_to_dict, binarize

# %%
# warnings.filterwarnings('ignore', module='matplotlib\backends\*')

log = logging.getLogger(__name__)

# %%
if False:
    # %%
    from hydra.experimental import compose, initialize_config_dir
    PROJECT_DIR='/home/jaesungyoo/EMG'
    PROJECT_DIR='/zdisk/jaesungyoo/EMG'
    os.chdir(os.path.join(PROJECT_DIR, 'train'))
    os.listdir()

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.join(PROJECT_DIR, 'conf'), job_name='test')
    EXP_DIR='multirun/train/2021-07-23/04-37-51/124'
    overrides = [
    # 'EXP_DIR=multirun/r8000/2021-05-24/11-39-32/20'
    f'EXP_DIR={EXP_DIR}'
    ]
    overrides = [
    'EXP_DIR=multirun/train2/Resnet/0',
    'score_v=2',
    'test=False',
    ]
    cfg = compose(config_name='eval_config', overrides=overrides, return_hydra_config=True)
    print(OmegaConf.to_yaml(cfg))
    os.chdir(os.path.join(PROJECT_DIR, 'train', cfg._EXP_DIR))
    os.getcwd()
    os.listdir()
    # Dummy class for debugging
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

# %%
def get_y_pred_frame(y_score, keys=list(range(3))):
    count = collections.Counter(y_score.argmax(axis=1))
    count = dict(count)
    for key in keys:
        if key not in count.keys():
            count[key]=0

    count = [count[key] for key in keys]
    return count

def pred_subject(y, y_score, y_encoder, test_cohort, save_dir=''):
    test_cohort['person_id'] = [filename.split('.')[0].split('_')[0] for filename in test_cohort['filename']]

    i = 0
    results = {'person_id':[], 'y':[], 'y_label':[], 'y_score':[], 'y_pred_frame':[]}
    for person_id, g in groupby(test_cohort[['NMNL', 'person_id']].values, key=lambda x: x[1]):
        l = list(g)
        n_frames = len(l)
        y_true, _ = l[0]

        y_ = y[i:i+n_frames]
        y_score_ = y_score[i:i+n_frames]
        assert T.numpy.equal(y_)
        assert T.numpy.equal(l)
        y_ = int(y_[0])
        assert y_true==y_encoder[y_], f'y_true: ({y_true}), y_encoder: ({y_encoder[y_]})'

        results['person_id'].append(person_id)
        results['y'].append(y_)
        results['y_label'].append(y_true)
        results['y_score'].append(y_score_.mean(axis=0))
        results['y_pred_frame'].append(get_y_pred_frame(y_score_))

        i += n_frames

    results_tosave = results.copy()
    results_tosave['y_score'] = list_to_dict(results_tosave['y_score'], keys=y_encoder)
    results_tosave['y_pred_frame'] = list_to_dict(results_tosave['y_pred_frame'], keys=y_encoder)
    results_tosave = T.unnest_dict(results_tosave)
    results_tosave = pd.DataFrame(results_tosave)
    results_tosave.to_csv(os.path.join(save_dir, 'subject_prediction.csv'), index=False)
    results_tosave.to_excel(os.path.join(save_dir, 'subject_prediction.xlsx'))

    results['y'] = np.array(results['y'])
    results['y_score'] = np.stack(results['y_score'], axis=0)

    return results

def pred_file(y, y_score, y_encoder, test_cohort):
    i=0
    results = {'wavefile_name': [], 'y': [], 'y_label':[], 'y_score': []}
    for wavefile_name, g in groupby(test_cohort[['NMNL', 'wavefile_name']].values, key=lambda x: x[1]):
        l = list(g)
        n_frames = len(l)
        y_true, _ = l[0]

        y_ = y[i:i+n_frames]
        y_score_ = y_score[i:i+n_frames]
        assert T.numpy.equal(y_)
        assert T.numpy.equal(l)
        y_ = int(y_[0])
        assert y_true==y_encoder[y_], f'y_true: ({y_true}), y_encoder: ({y_encoder[y_]})'

        results['wavefile_name'].append(wavefile_name)
        results['y'].append(y_)
        results['y_label'].append(y_true)
        results['y_score'].append(y_score_.mean(axis=0))
        # results['y_pred_frame'].append(get_y_pred_frame(y_score_))

        i += n_frames

    results['y'] = np.array(results['y'])
    results['y_score'] = np.stack(results['y_score'], axis=0)

    return results

def refine_cohort(test_cohort, exp_cfg):
    if hasattr(exp_cfg, 'PROJECT_DIR'):
        data_src_dir = os.path.join(exp_cfg.PROJECT_DIR, 'data/data', exp_cfg.encoding.run_dir)
    elif hasattr(exp_cfg, 'path'):
        data_src_dir = os.path.join(exp_cfg.path.project_dir, 'data/data', exp_cfg.encoding.run_dir)
    else:
        data_src_dir = os.path.join(exp_cfg.path_root, 'data/data', exp_cfg.encoding.run_dir)
    data_src_path = T.Path(data_src_dir)
    data_src_path.WAV = 'wav'

    person_id_list = []
    wavefile_name_list = []
    wavefile_dir_list = []
    for filename in test_cohort['filename']:
        filename_parts = filename.split('.')[0].split('_') # 123456_0_0.p -> [123456, 0, 0]
        wavefile_name = '_'.join(filename_parts[:2])
        wavefile_dir = wavefile_name+'.p'
        person_id_list.append(int(filename_parts[0]))
        wavefile_name_list.append(wavefile_name)
        wavefile_dir_list.append(data_src_path.WAV.join(wavefile_dir))
    test_cohort['person_id'] = person_id_list
    test_cohort['wavefile_name'] = wavefile_name_list
    test_cohort['wavefile_dir'] = wavefile_dir_list

    return test_cohort

# %%
def add_triangle(ax):
    x_,y_,z_=np.eye(3)
    verts = [list(zip(x_,y_,z_))]
    srf = Poly3DCollection(verts, alpha=0.25, facecolor='k')
    ax.add_collection3d(srf)

def set_labels_lims(ax, y_encoder):
    ax.set_xlim([0,1])
    ax.set_ylim([1,0])
    ax.set_zlim([0,1])
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(f'P({y_encoder[0]})')
    ax.set_ylabel(f'P({y_encoder[1]})')
    ax.set_zlabel(f'P({y_encoder[2]})')

# %%
class Scorer():
    def __init__(self, RESULT_DIR=None):
        self.RESULT_DIR=T.Path(RESULT_DIR)
        if self.RESULT_DIR is not None:
            os.makedirs(self.RESULT_DIR, exist_ok=True)
            os.makedirs(self.RESULT_DIR.join('plot'), exist_ok=True)

    def __call__(self, y, y_score, y_encoder, test_cohort, x=None, exp_cfg=None, plot=True):
        # Refine cohort
        test_cohort = refine_cohort(test_cohort, exp_cfg)
        # Score frame
        log.info('[Score: frame]')
        result_frame = self.score_frame(y=y, y_score=y_score, y_encoder=y_encoder, test_cohort=test_cohort, x=x, exp_cfg=exp_cfg, plot=plot)
        # Score subject
        log.info('[Score: subject]')
        result_subject = self.score_subject(y=y, y_score=y_score, y_encoder=y_encoder, test_cohort=test_cohort, x=x, exp_cfg=exp_cfg, plot=plot)
        # Merge scores
        result_all = {
        'frame': result_frame,
        'subject': result_subject,
        }

        return result_all

    def score_frame(self, y, y_score, y_encoder, test_cohort=None, x=None, exp_cfg=None, plot=True):
        os.makedirs(self.RESULT_DIR.join('frame'), exist_ok=True)
        return self.scores(y, y_score, y_encoder, savename='result_frame.p', x=x, exp_cfg=exp_cfg, save_dir=self.RESULT_DIR.join('frame'), plot=plot)

    def score_subject(self, y, y_score, y_encoder, test_cohort, x=None, exp_cfg=None, plot=True):
        os.makedirs(self.RESULT_DIR.join('subject'), exist_ok=True)
        if plot:
            self.plot_prediction(y=y, y_score=y_score, y_encoder=y_encoder, test_cohort=test_cohort, exp_cfg=exp_cfg)
            self.plot_heatmap(y=y, y_score=y_score, y_encoder=y_encoder, test_cohort=test_cohort, exp_cfg=exp_cfg)
        results = pred_subject(y, y_score, y_encoder, test_cohort, save_dir=self.RESULT_DIR)

        y = np.array(results['y'])
        y_score = np.array(results['y_score'])
        # Save y, y_score
        pred_results_subject = {'y':y, **{f'y_score.{i}':y_score_ for i, y_score_ in enumerate(y_score.T)}}
        pred_results_subject = pd.DataFrame(pred_results_subject)
        pred_results_subject.to_csv(self.RESULT_DIR.join('pred_results_subject.csv'))
        # content = f'y:\n{y}\ny_score:\n{y_score}'
        # T.write(content, self.RESULT_DIR.join('y_y_score.txt'))

        if plot:
            self.plot_featmap(y=y, y_score=y_score, y_encoder=y_encoder)

        return self.scores(y, y_score, y_encoder, savename='result_subject.p', exp_cfg=exp_cfg, save_dir=self.RESULT_DIR.join('subject'), plot=plot)

    def score_subject2(self, model_skl, result_train, result_test, y_encoder, cfg):
        '''Pass model to plot respose curve'''
        os.makedirs(self.RESULT_DIR.join('subject2'), exist_ok=True)
        ex_type = cfg.subfeatextract.ex_type

        x, y_true, y_score = result_train['x'], result_train['y_true'], result_train['y_score']
        self.plot_featmap(y=y_true, y_score=y_score, y_encoder=y_encoder, suffix='_train_embedded') # encoded features
        if ex_type=='all':
            self.plot_response(model_skl=model_skl, x=x, y_encoder=y_encoder, suffix='_train') # response to train dataset
            self.plot_response(model_skl=model_skl, y_encoder=y_encoder) # response curve of model_skl
            self.plot_featmap(y=y_true, y_score=x, y_encoder=y_encoder, suffix='_train_data', title_suffix=' (train_data)') # original features

        # Show response to Testing results
        x, y_true, y_score = result_test['x'], result_test['y_true'], result_test['y_score']
        self.plot_featmap(y=y_true, y_score=y_score, y_encoder=y_encoder, suffix='_test_embedded')
        if ex_type=='all':
            self.plot_response(model_skl=model_skl, x=x, y_encoder=y_encoder, suffix='_test')
            self.plot_featmap(y=y_true, y_score=x, y_encoder=y_encoder, suffix='_test_data', title_suffix=' (test_data)')

        scores = self._score_subject2(y=y_true, y_score=y_score, y_encoder=y_encoder, exp_cfg=cfg)

        # Save y, y_score
        # pred_results_subject = pd.DataFrame({'y': y, 'y_score': y_score})
        # pred_results_subject.to_csv(self.RESULT_DIR.join('pred_results_subject.csv'))

        # content = f'y:\n{y}\ny_score:\n{y_score}'
        # T.write(content, self.RESULT_DIR.join('y_y_score.txt'))
        # T.save_pickle({'y': y, 'y_score': y_score}, 'y_y_score.p')

        # result = self.scores(y, y_score, y_encoder, savename='result_subject.p', exp_cfg=cfg, save_dir=self.RESULT_DIR.join('subject2'))
        # pp=pprint.PrettyPrinter(indent=2)
        # content = pp.pformat(result)
        #
        # log.info('\n'+content)
        # T.write(content, self.RESULT_DIR.join('result.txt'))

        return scores

    def _score_subject2(self, y, y_score, y_encoder, exp_cfg=None):

        # Save y, y_score
        pred_results_subject = {'y':y, **{f'y_score.{i}':y_score_ for i, y_score_ in enumerate(y_score.T)}}
        pred_results_subject = pd.DataFrame(pred_results_subject)
        pred_results_subject.to_csv(self.RESULT_DIR.join('pred_results_subject.csv'))

        # content = f'y:\n{y}\ny_score:\n{y_score}'
        # T.write(content, self.RESULT_DIR.join('y_y_score.txt'))
        # T.save_pickle({'y': y, 'y_score': y_score}, 'y_y_score.p')

        scores = self.scores(y, y_score, y_encoder, savename='result_subject.p', exp_cfg=exp_cfg, save_dir=self.RESULT_DIR.join('subject2'))
        pp=pprint.PrettyPrinter(indent=2)
        content = pp.pformat(scores)

        log.info('\n'+content)
        T.write(content, self.RESULT_DIR.join('scores.txt'))

        return scores

    def scores(self, y, y_score, y_encoder, savename='result.p', x=None, exp_cfg=None, save_dir=None, plot=True):

        results_ovo = self.metric_ovo(y, y_score, y_encoder, plot=plot)
        results_ovr = self.metric_ovr(y, y_score, y_encoder, plot=plot)
        results_all = self.metric_all(y, y_score, save_dir=save_dir)
        if plot:
            if x is not None:
                if x.ndim==2:
                    if 'sr_trg' in exp_cfg.method:
                        sr = exp_cfg.method.sr_trg
                    else:
                        sr = exp_cfg.encoding.sr
                    log.info('plotting sliced wavefiles named after its prediction...')
                    self.plot_pred(x, y, y_score, y_encoder, sr)
                elif x.ndim==3 or x.ndim==4:
                    log.info('plotting sliced wave images named after its prediction...')
                    self.plot_pred2d(x, y, y_score, y_encoder)

        results = {}
        for (c1, c2), scores in results_ovo.items():
            results[f'{y_encoder[c1]}_{y_encoder[c2]}'] = scores
        for c, scores in results_ovr.items():
            results[f'{y_encoder[c]}'] = scores
        results['all'] = results_all
        T.save_pickle(results, self.RESULT_DIR.join(savename))

        return results

    def metric_ovo(self, y_true, y_score, y_encoder=None, plot=False):
        '''One-vs-One
        y_true: shape of (n_samples)
        y_score: shape of (n_samples, n_clsses)'''
        classes = list(range(y_score.shape[1]))
        # classes = list(set(y_true).union(set(y_pred)))

        results = {}
        for c1, c2 in combinations(classes, 2):
            '''False: c1, True: c2'''
            # Crop used classes
            y_true_c1 = y_true==c1
            y_true_c2 = y_true==c2
            crop_i = np.logical_or(y_true_c1, y_true_c2)

            y_true_ = y_true_c2[crop_i]
            y_score_ = y_score[:, [c1,c2]]
            y_score_ = y_score_[crop_i]
            y_score_ = y_score_[:, 1] / y_score_.sum(axis=1) # normalize
            y_score_ = np.nan_to_num(y_score_, nan=0.5) # impute 0 values with 0.5

            if plot:
                save_dir = self.RESULT_DIR.join(f'{y_encoder[c1]}_{y_encoder[c2]}') if y_encoder != None else self.RESULT_DIR.join(f'{c1}_{c2}')
                results[c1,c2] = self.metric_binary(y_true_, y_score_, save_dir=save_dir)
            else:
                results[c1,c2] = self.metric_binary(y_true_, y_score_)
        return results

    def metric_ovr(self, y_true, y_score, y_encoder=None, plot=False):
        '''One-vs-Rest
        y_true: shape of (n_samples)
        y_score: shape of (n_samples, n_clsses)'''
        classes = list(range(y_score.shape[1]))
        # classes = list(set(y_true).union(set(y_score)))

        results = {}
        for c in classes:
            y_true_ = y_true==c
            y_score_ = y_score[:, c]

            if plot:
                save_dir = self.RESULT_DIR.join(f'{y_encoder[c]}') if y_encoder != None else self.RESULT_DIR.join(f'{c}')
                results[c] = self.metric_binary(y_true_, y_score_, save_dir=save_dir)
            else:
                results[c] = self.metric_binary(y_true_, y_score_)

        return results

    def metric_all(self, y_true, y_score, save_dir=None):
        y_pred = np.argmax(y_score, axis=1)

        auroc = roc_auc_score(y_true, y_score, multi_class='ovr')

        result = {'y_true': y_true, 'y_pred': y_pred}
        classification_result = hs.E.classification_report_full(result, discard_ovr=True)
        # accuracy = accuracy_score(y_true, y_pred)
        # f1_all = f1_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)
        m_cc = matthews_corrcoef(y_true, y_pred)
        c_matrix = confusion_matrix(y_true, y_pred)

        if save_dir is not None:
            df_cm = pd.DataFrame(c_matrix)
            fig = plt.figure()
            ax = sns.heatmap(df_cm, annot=True)
            ax.set_ylabel('y_true')
            ax.set_xlabel('y_pred')
            fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close(fig)

        scores = {
        'auroc': auroc,
        # 'accuracy': accuracy,
        # 'f1': f1_all,
        'kappa': kappa,
        'm_cc': m_cc,
        'c_matrix': c_matrix,
        }
        classification_result.update(scores)
        return classification_result
        # return scores

    def metric_binary(self, y_true, y_score, save_dir=None, suffix=''):
        if save_dir != None:
            os.makedirs(save_dir, exist_ok=True)

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
        fig, ax = plt.subplots()
        ax.set_title('precision_recall_curve')
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.plot(recalls, precisions)
        if save_dir != None:
            # fig.savefig(os.path.join(save_dir, f'pr_curve{suffix}.eps'))
            fig.savefig(os.path.join(save_dir, f'pr_curve{suffix}.svg'))
            fig.savefig(os.path.join(save_dir, f'pr_curve{suffix}.png'))
        plt.close(fig)

        threshold_i = np.where(recalls>=0.8)[0]
        if len(threshold_i)==0:
            # Threshold lowest
            # precision = P / (P+F)
            # recall = 1
            threshold_i = None
            threshold = 0
            precision = precision_score(y_true, np.ones_like(y_true))
            recall = 1
        else:
            # Threshold highest
            # precision = 1
            # recall = 0
            threshold_i = threshold_i[-1]
            threshold = thresholds[threshold_i]
            precision = precisions[threshold_i]
            recall = recalls[threshold_i]

        auroc = roc_auc_score(y_true, y_score)

        y_pred = binarize(y_score.reshape(-1,1), threshold=threshold)
        c_matrix = confusion_matrix(y_true, y_pred)
        if save_dir is not None:
            c_matrix = confusion_matrix(y_true, y_pred)
            df_cm = pd.DataFrame(c_matrix)
            fig = plt.figure()
            ax = sns.heatmap(df_cm, annot=True)
            ax.set_ylabel('y_true')
            ax.set_xlabel('y_pred')
        '''
        [[tn, fp],
        [fn, tp]]
        '''

        assert recall_score(y_true, y_pred) == recall
        assert precision_score(y_true, y_pred) == precision
        assert recall>=0.8

        accuracy = accuracy_score(y_true, y_pred)
        f1=f1_score(y_true, y_pred)

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fig, ax = plt.subplots()
        ax.set_title('roc curve')
        ax.set_xlabel('fpr')
        ax.set_ylabel('tpr')
        ax.plot(fpr, tpr)
        if save_dir != None:
            # fig.savefig(os.path.join(save_dir, f'roc_curve{suffix}.eps'))
            fig.savefig(os.path.join(save_dir, f'roc_curve{suffix}.svg'))
            fig.savefig(os.path.join(save_dir, f'roc_curve{suffix}.png'))
        plt.close(fig)

        # Plot histogram of y_score
        fig, ax = plt.subplots()
        ax.set_title('$\hat{y}$ distribution')
        ax.set_xlabel('P')
        ax.set_ylabel('N')
        ax.hist(y_score, bins=50, range=(0,1))
        if save_dir != None:
            # fig.savefig(os.path.join(save_dir, f'yhat_distrib{suffix}.eps'))
            fig.savefig(os.path.join(save_dir, f'yhat_distrib{suffix}.svg'))
            fig.savefig(os.path.join(save_dir, f'yhat_distrib{suffix}.png'))
        plt.close(fig)

        scores = {
        'accuracy': accuracy,
        'f1': f1,
        'auroc': auroc,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'c_matrix': c_matrix,
        }
        return scores

    def plot_pred(self, x, y_true, y_score, y_encoder, sr):
        for y_val in y_encoder:
            os.makedirs(self.RESULT_DIR.join(f'plot/{y_val}'), exist_ok=True)

        y_pred = np.argmax(y_score, axis=1)
        n_y = np.zeros((len(y_encoder),len(y_encoder)), dtype=int)

        for x_, y_true_, y_pred_ in zip(x, y_true, y_pred):
            n_y[y_true_, y_pred_] += 1

            fig = plt.figure()
            ax = fig.gca()
            ax.plot(x_)
            ax.set_ylim([6,-6])
            sec = len(x_) / sr
            xticks = np.linspace(0,sec,len(ax.get_xticklabels())-2)
            xticks = np.round(xticks, 3)
            d = xticks[1]-xticks[0]
            xticks = np.concatenate([[xticks[0] - d],xticks,[xticks[-1]+d]])
            ax.set_xticklabels(xticks)

            # fig.savefig(self.RESULT_DIR.join(f'plot/{y_encoder[y_true_]}/{y_encoder[y_pred_]}_{n_y[y_true_, y_pred_]}.eps'))
            fig.savefig(self.RESULT_DIR.join(f'plot/{y_encoder[y_true_]}/{y_encoder[y_pred_]}_{n_y[y_true_, y_pred_]}.svg'))
            fig.savefig(self.RESULT_DIR.join(f'plot/{y_encoder[y_true_]}/{y_encoder[y_pred_]}_{n_y[y_true_, y_pred_]}.png'))
            plt.close(fig)

    def plot_pred2d(self, x, y_true, y_score, y_encoder):
        for y_val in y_encoder:
            os.makedirs(self.RESULT_DIR.join(f'plot/{y_val}'), exist_ok=True)

        y_pred = np.argmax(y_score, axis=1)
        n_y = np.zeros((len(y_encoder),len(y_encoder)), dtype=int)

        for x_, y_true_, y_pred_ in zip(x, y_true, y_pred):
            n_y[y_true_, y_pred_] += 1

            if x_.ndim==3:
                x_ = x_.mean(axis=0) # Mean accross pytorch channel dimension
                # x_ = np.transpose(x_, axes=(1,2,0))

            fig = plt.figure()
            ax = fig.gca()
            img = ax.matshow(x_, cmap='magma')
            fig.colorbar(img)

            # fig.savefig(self.RESULT_DIR.join(f'plot/{y_encoder[y_true_]}/{y_encoder[y_pred_]}_{n_y[y_true_, y_pred_]}.eps'))
            fig.savefig(self.RESULT_DIR.join(f'plot/{y_encoder[y_true_]}/{y_encoder[y_pred_]}_{n_y[y_true_, y_pred_]}.svg'))
            fig.savefig(self.RESULT_DIR.join(f'plot/{y_encoder[y_true_]}/{y_encoder[y_pred_]}_{n_y[y_true_, y_pred_]}.png'))
            plt.close(fig)

    def plot_prediction(self, y, y_score, y_encoder, test_cohort, exp_cfg):
        # shutil.rmtree('result/plot/heatmap', ignore_errors=True)
        os.makedirs(self.RESULT_DIR.join('plot','prediction'), exist_ok=True)
        downsample = 'sr_trg' in exp_cfg.method
        if downsample:
            sr = exp_cfg.method.sr_trg
        else:
            sr = exp_cfg.encoding.sr
        window_T = exp_cfg.method.window * sr
        edge_T = window_T / 2

        def plot(wav, x, y_score, title, wavefile_name):
            fig, ax = plt.subplots(nrows=2,figsize=(14,5), gridspec_kw={'height_ratios':[4,3]})

            fig.suptitle(title)
            # Plot signal
            ax[0].set_title('waveform')
            ax[0].plot(wav, color='black')
            ax[0].set_ylim([-6,6])
            ax[0].set_ylabel('amplitude [mV]')
            xlim = ax[0].get_xlim()

            # Color y_score
            ax[0].set_title('heatmap')
            for x_, y_score_, y_loc in zip(x, y_score, np.linspace(5,-5,len(y_score))):
                ax[0].plot(x_, y_loc, color=tuple(y_score_), marker='o')
                ax[0].plot([x_ - edge_T, x_ + edge_T], [y_loc, y_loc], color=tuple(y_score_)+(0.5,), linestyle='--')
                ax[0].plot([x_ - edge_T, x_ + edge_T], [y_loc, y_loc], color=tuple(y_score_)+(0.5,), marker='|')

            # Plot y_score
            ax[1].set_title('y_score')
            ax[1].plot(x, y_score[:,0], color=(1,0,0), marker='o', linestyle='-')
            ax[1].plot(x, y_score[:,1], color=(0,1,0), marker='o', linestyle='-')
            ax[1].plot(x, y_score[:,2], color=(0,0,1), marker='o', linestyle='-')
            ax[1].set_xlim(xlim)
            ax[1].set_ylim([-0.05,1.05])
            ax[1].set_ylabel('probability')
            ax[1].legend(y_encoder)

            # fig.savefig(self.RESULT_DIR.join(f'plot/prediction/{wavefile_name}.eps'))
            fig.savefig(self.RESULT_DIR.join(f'plot/prediction/{wavefile_name}.svg'))
            fig.savefig(self.RESULT_DIR.join(f'plot/prediction/{wavefile_name}.png'))
            plt.close(fig)

        log.info('[Plotting prediction]')
        i = 0
        gb=groupby(test_cohort.iloc, key=lambda x: x['wavefile_name'])
        for wavefile_name, group in gb:
            log.info(wavefile_name)
            row_list = list(group)
            row = row_list[0]
            n_window = len(row_list)
            i_end = i + n_window
            NMNL = row['NMNL']
            muscle = row['muscle']

            x = [int(row['filename'].split('.')[0].split('_')[-1]) for row in row_list]
            x = (np.array(x) + sr * exp_cfg.method.window /2).astype(int) # add half window size to center to middle of the window
            wav = T.load_pickle(row['wavefile_dir'])
            if downsample:
                T_wav = len(wav)
                s = T_wav/exp_cfg.encoding.sr
                T_hat = int(s * sr)
                wav = signal.resample(wav, T_hat)
            y_score_ = y_score[i:i_end]
            title = f'{wavefile_name} / {NMNL} / {muscle}'

            plot(wav, x, y_score_, title, wavefile_name)
            i = i_end

        assert i==len(y_score), f'i: {i}, len(y_score): {len(y_score)}'

    def plot_heatmap(self, y, y_score, y_encoder, test_cohort, exp_cfg):
        def plot_legend(ax=None, labels=['A','B','C']):
            ''' Plots a legend for the colour scheme
            given by abc_to_rgb. Includes some code adapted
            from http://stackoverflow.com/a/6076050/637562'''

            # Basis vectors for triangle
            basis = np.array([[0.0, 1.0], [-1.5/np.sqrt(3), -0.5],[1.5/np.sqrt(3), -0.5]])

            if ax==None:
                fig = plt.figure()
                ax = fig.add_subplot(111,aspect='equal')

            # Plot points
            a, b, c = np.mgrid[0:1:2e-2, 0:1:2e-2, 0:1:2e-2]
            a, b, c = a.flatten(), b.flatten(), c.flatten()
            abc = np.stack((a,b,c), axis=1)
            abc = abc / abc.sum(axis=1)[...,None]
            data = abc @ basis
            c = abc
            ax.scatter(data[:,0], data[:,1],marker='^', s=30, edgecolors='none',facecolors=c)

            offset = 0.25
            fontsize = 16
            ax.text(basis[0,0]*(1+offset), basis[0,1]*(1+offset), labels[0], horizontalalignment='center',
                    verticalalignment='center', fontsize=fontsize)
            ax.text(basis[1,0]*(1+offset), basis[1,1]*(1+offset), labels[1], horizontalalignment='center',
                    verticalalignment='center', fontsize=fontsize)
            ax.text(basis[2,0]*(1+offset), basis[2,1]*(1+offset), labels[2], horizontalalignment='center',
                    verticalalignment='center', fontsize=fontsize)

            ax.set_frame_on(False)
            ax.set_xticks(())
            ax.set_yticks(())


        def plot(heatmap):
            shape = heatmap.shape[:2][::-1]
            shape = (int(shape[0]), int(shape[1]/4))
            fig = plt.figure(figsize=shape)
            gs = fig.add_gridspec(3,2, width_ratios=[5,1])
            ax1 = fig.add_subplot(gs[:,0])
            ax2 = fig.add_subplot(gs[1,1])

            # heatmap
            ax1.imshow(heatmap, aspect='auto')
            ax1.set_xticks(range(n_patient))
            ax1.set_xticklabels([person_id_outcome[person_id] for person_id in person_id_i.keys()])
            ax1.set_xlabel('Outcome')
            secxax1 = ax1.secondary_xaxis('top')
            secxax1.set_xticks(range(n_patient))
            secxax1.set_xticklabels(person_id_i.keys(), fontsize=6)
            secxax1.set_xlabel('person_id')
            ax1.set_yticks(range(n_muscle))
            ax1.set_yticklabels(muscle_i.keys())
            ax1.set_ylabel('Muscle type')
            secyax1 = ax1.secondary_yaxis('right')
            secyax1.set_yticks(range(n_muscle))
            secyax1.set_yticklabels([muscle_PD_PD[muscle] for muscle in muscle_i.keys()])
            secyax1.set_ylabel('Proximal / Distal')

            # Triangular colormap
            labels = y_encoder
            # plot_legend(ax2, labels)
            # ax2.set_aspect('equal')
            return fig

        test_cohort['muscle_PD'] = test_cohort['muscle_crop'] + '('+test_cohort['PD']+')'
        n_patient = test_cohort['person_id'].nunique()
        n_muscle = test_cohort['muscle_PD'].nunique()
        person_id_outcome = dict(test_cohort[['person_id', 'NMNL']].value_counts().index)
        person_id_i = {person_id_outcome_[0]:i for i, person_id_outcome_ in enumerate(sorted(person_id_outcome.items(), key=lambda x: x[1]))} # sort by NMNL
        muscle_PD = pd.DataFrame(np.stack(test_cohort[['muscle_crop', 'PD']].value_counts().index, axis=0), columns=['muscle_crop', 'PD']).sort_values(by=['PD','muscle_crop'], ascending=[False,True])
        muscle_PD = muscle_PD['muscle_crop'] + '('+muscle_PD['PD']+')'
        muscle_PD = list(muscle_PD.values)
        muscle_PD_PD = {muscle:muscle[-2] for muscle in  muscle_PD}

        # muscle_PD = {m_PD[0]: m_PD[1] for m_PD in muscle_PD.values}
        # muscle_i = {muscle:i for i, muscle in enumerate(muscle_PD.keys())}
        # muscle_PD = [(m_PD[0], m_PD[1]) for m_PD in muscle_PD.values]
        muscle_i = {muscle:i for i, muscle in enumerate(muscle_PD)}
        heatmap = np.ones((n_muscle, n_patient, 3))

        log.info('[Plotting heatmap]')
        i = 0
        gb=groupby(test_cohort.iloc, key=lambda x: x['wavefile_name'])
        for wavefile_name, group in gb:
            log.info(wavefile_name)
            row_list = list(group)
            row = row_list[0]
            n_window = len(row_list)
            i_end = i + n_window
            person_id = row['person_id']
            NMNL = row['NMNL']
            muscle = row['muscle_PD']

            x = [int(row['filename'].split('.')[0].split('_')[-1]) for row in row_list]
            x = (np.array(x) + exp_cfg.encoding.sr * exp_cfg.method.window /2).astype(int) # add half window size to center to middle of the window
            wav = T.load_pickle(row['wavefile_dir'])
            y_score_ = y_score[i:i_end]
            y_score_avg = y_score_.mean(axis=0)

            heatmap[muscle_i[muscle], person_id_i[person_id]] = y_score_avg

            i = i_end

        fig = plot(heatmap)
        # fig.savefig(self.RESULT_DIR.join(f'plot/heatmap.eps'))
        fig.savefig(self.RESULT_DIR.join(f'plot/heatmap.svg'))
        fig.savefig(self.RESULT_DIR.join(f'plot/heatmap.png'))
        plt.close(fig)

    def plot_featmap(self, y, y_score, y_encoder, save=True, title_suffix='', suffix=''):
        '''
        Plot features in 3-dimensional triangular plane
        :param y_score: array of (N, 3)
        '''
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Scatter plot
        colors = ['r', 'g', 'b']
        for y_i, (y_, c) in enumerate(zip(y_encoder, colors)):
            y_score_ = y_score[y==y_i] # Get y_score which labels are y_

            # ax.scatter(*y_score_.T, label=y_)
            ax.scatter(y_score_[:,0], y_score_[:,1], y_score_[:,2], color=c, label=y_)

        # plot triangle surface x+y+z=1
        x,y,z=np.eye(3)-1e-5
        verts = [list(zip(x,y,z))]
        srf = Poly3DCollection(verts, alpha=0.25, facecolor='k')
        ax.add_collection3d(srf)

        ax.set_xlabel(y_encoder[0])
        ax.set_ylabel(y_encoder[1])
        ax.set_zlabel(y_encoder[2])
        ax.set_xlim([0,1])
        ax.set_ylim([1,0])
        ax.set_zlim([0,1])
        ax.set_title(f'featmap{title_suffix}')
        ax.legend()

        if save:
            T.plot.multisave(fig, self.RESULT_DIR.join(f'plot/featmap{suffix}'))
            plt.close(fig)
        return fig, ax

    def plot_decision_boundary(self, model_skl, y_encoder, save=True):
        fig, ax = plt.subplots()

    def plot_response(self, model_skl, y_encoder, x=None, save=True, title_suffix='', suffix=''):
        if False:
            self.plot_response(model_skl=model_skl, x=x, y_encoder=y_encoder, suffix='_train') # response to train dataset
            model_skl=model2
            x=None
            x = result_train['x']

        if x is None:
            x_, y_, z_ = np.mgrid[0:1:1e-2, 0:1:1e-2, 0:1:1e-2]
            data = np.stack((d.flatten() for d in (x_,y_,z_)), axis=1).astype(np.float32)
            vis_i = (0.9 <= data.sum(axis=1)) & (data.sum(axis=1) <=1) # indices to save
            data = data[vis_i]
            # data = torch.as_tensor(data, dtype=torch.float32)
        else:
            # Assume numpy array
            data = x

        y_score = model_skl.predict_proba(data)
        # data = data.to(T.torch.get_device(model))
        # with torch.no_grad():
        #     y_score = torch.softmax(model(data), dim=1).cpu().numpy()
        # data = data.cpu().numpy()
        targets_pred = y_score.argmax(axis=1)
        targets_pred_ = targets_pred

        c_list = ['r', 'g', 'b']
        c_rgb = np.stack(tuple(mcolors.to_rgb(c) for c in c_list), axis=0) # Convert r,g,b into rgb values
        color_score = y_score @ c_rgb
        # c=np.stack([(c_rgb * y_score_[...,None]).sum(axis=0) for y_score_ in y_score], axis=0)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection='3d')

        # Add triangle x+y+z=1
        if x is not None:
            add_triangle(ax1)

        x_,y_,z_ = data.T
        ax1.scatter(x_, y_, z_, c=color_score, depthshade=False)
        set_labels_lims(ax1, y_encoder)
        ax1.set_title('Response_score (position is input, color is the output)')
        if save:
            T.plot.multisave(fig1, self.RESULT_DIR.join(f'plot/responsemap_score{suffix}'))
            plt.close(fig1)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection='3d')
        # Add triangle
        if x is not None:
            add_triangle(ax2)

        for i, c in zip(range(3), c_list):
            target_i = targets_pred_==i
            x_,y_,z_ = data[target_i].T
            scatter = ax2.scatter(x_,y_,z_, color=c, depthshade=False, edgecolors='k')
        set_labels_lims(ax2, y_encoder)
        ax2.set_title('Response_pred (position is input, color is the output)')
        if save:
            T.plot.multisave(fig2, self.RESULT_DIR.join(f'plot/responsemap_pred{suffix}'))
            plt.close(fig2)
        return [fig1, fig2], [ax1, ax2]

# %%
def intmodel(model_dir):
    '''model_dir: "model_10.pt"'''
    try:
        x=int(model_dir.split('.')[0].split('_')[-1])
        return True
    except ValueError:
        return False

def get_best_model(model_dir):
    model_list = [model_dir for model_dir in os.listdir(model_dir) if intmodel(model_dir)]
    model_list = sorted(model_list, key=lambda x: int(x.split('.')[0].split('_')[1]))
    best_model = model_list[-1]
    return best_model

def get_model(path, model_type):
    if model_type=='best':
        return get_best_model(path)
    elif model_type=='final':
        return 'model_final.pt'
    elif type(model_type)==int:
        return f'model_{model_type}.pt'
    else:
        raise Exception(f"model_type must be either ['best', 'final', int], received: {model_type}")

def version_check(model_cfg):
    # CNN1, CNN2 deprecated. Replace with CNN3 & CNN4
    model_cfg = dcopy(model_cfg)
    if model_cfg._target_ == 'model.CNN1':
        model_cfg._target_ = 'model.CNN3'
    elif model_cfg._target_ == 'model.CNN2':
        model_cfg._target_ = 'model.CNN4'
    return model_cfg

def load_state_dict(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if 'layers.18._n_input' in state_dict.keys():
        del state_dict['layers.18._n_input']
    if 'layers.18._fit' in state_dict.keys():
        del state_dict['layers.18._fit']
    return state_dict

# def parse_y_y_score(y_y_score):
#
#     for line in y_y_score:
#         line.replace('\n', '')
#         if line=='y:':
#             pass
#
#     return y, y_score

# %%
@hydra.main(config_path='../conf', config_name='eval_config')
def main(cfg: DictConfig) -> None:
    log.info('[eval_cfg]')
    log.info(OmegaConf.to_yaml(cfg))

    # Set GPU for current experiment
    device = T.torch.get_device(gpu_id=cfg.gpu_id)
    log.info(device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    log.info('-'*30)
    log.info(os.getcwd())
    exp_path = T.Path()
    exp_path.MODEL = 'model/'

    exp_cfg = OmegaConf.load('.hydra/config.yaml')
    exp_cfg.PROJECT_DIR = cfg.PROJECT_DIR
    exp_cfg.path_root = cfg.PROJECT_DIR
    log.info('[exp_cfg]')
    log.info(OmegaConf.to_yaml(exp_cfg))

    assert cfg.score_v in [1,2]

    if cfg.test:
        path = T.Path(cfg.PROJECT_DIR)
        path.DATA = os.path.join('data', exp_cfg.DATA_DIR)
        path.DATA.WAV = 'wav'
        path.DATA.cohort = 'cohort.csv'

        cohort = pd.read_csv(path.DATA.cohort)
    # cohort['filename']=cohort['filename']+'.p'
        test_i = T.load_pickle(exp_path.join('test_i.p'))
        test_cohort = cohort.iloc[test_i]
    # cohort
    # test_cohort
        # Testing
        def get_dataset(wav_dir, cohort, exp_cfg):
            if exp_cfg.torch_dataset._target_=='dataset.SimpleEMGDataset':
                dataset = hydra.utils.instantiate(exp_cfg.torch_dataset, wav_dir = wav_dir, cohort=cohort,
                                        x_channel=exp_cfg.x_channel, y_array=exp_cfg.y_array)
            elif exp_cfg.torch_dataset._target_=='dataset.MelDataset':
                dataset = hydra.utils.instantiate(exp_cfg.torch_dataset, wav_dir=wav_dir, cohort=cohort, sr=exp_cfg.method.sr_trg, scale=False, hop_length=exp_cfg.hop_length)
            else:
                raise Exception(f'Unknown dataset: {exp_cfg.torch_dataset}')
            return dataset
        test_dataset = get_dataset(wav_dir=path.DATA.WAV, cohort=test_cohort, exp_cfg=exp_cfg)
        test_loader = D.DataLoader(test_dataset, batch_size=exp_cfg.batch_size, shuffle=False)

        exp_cfg.exp.model = version_check(exp_cfg.exp.model) # Clean up the mess
        model = hydra.utils.instantiate(exp_cfg.exp.model)

        # Best model
        log.info('[Testing: best model]')
        model_filename = get_model(exp_path.MODEL, 'best')
        log.info(f'loading from: [{exp_path.MODEL.join(model_filename)}]')
        state_dict = load_state_dict(exp_path.MODEL.join(model_filename)) # Clean up the mess
        model.load_state_dict(state_dict)
        model.to(device)

        result = test(model, test_loader)
        x, y, y_score = result['x'], result['y'], result['y_score']
    else:
        y_y_score = T.readlines(os.path.join('result','best','y_y_score.txt'))
        y, y_score = parse_y_y_score(y_y_score)
        x=None

    if cfg.score_v==1:
        y_encoder = test_loader.dataset.y_encoder.keys
        scorer = Scorer(RESULT_DIR=os.path.join('result','best'))
        scores = scorer(x=x, y=y, y_score=y_score, y_encoder=y_encoder, test_cohort=test_cohort, exp_cfg=exp_cfg, plot=cfg.plot)
    elif cfg.score_v==2:
        scorer = Scorer(RESULT_DIR=os.path.join('result', 'best'))
        scorer._score_subject2(y=y, y_score=y_score, y_encoder=y_encoder, test_cohort=test_cohort, x=x, exp_cfg=exp_cfg)

# %%
if __name__ == '__main__':
    main()


if False:
    # %%
    import os
    os.chdir('/home/EMG/train')
    from hydra.experimental import compose, initialize_config_dir
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize_config_dir(config_dir='/home/EMG/conf', job_name='eval_config')
    cfg = compose(config_name='eval_config', return_hydra_config=True, overrides=['path_root=${hydra.runtime.cwd}', 'job_num=${hydra.job.num}'])
    print(OmegaConf.to_yaml(cfg))
    cfg.path_root='/home/EMG'
    cfg._EXP_DIR = 'multirun/2021-04-20/03-25-10/18'
    os.chdir(cfg._EXP_DIR)

    exp_cfg.exp.model._target_='model.CNN3'
    # model.load_state_dict(state_dict)
    model.layers[18].layers[1].weight.shape

    exp_cfg.exp.model
    state_dict
