# %%
import itertools as it
import logging
import os
import pprint
import shutil
from pathlib import Path

# %%
import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd

import dataset as D
import eval as E
import hideandseek as hs
import tools as T
import utils as U

# %%
log = logging.getLogger(__name__)

# %%
if False:
    # %%
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=os.path.join(os.getcwd(),'conf'), job_name='train')
    overrides = []
    cfg = hydra.compose(config_name='train', overrides=overrides)
    print(OmegaConf.to_yaml(cfg))
    log.info=print

# %%
@hydra.main(config_path='../conf', config_name='train', version_base='1.2')
def main(cfg: DictConfig) -> None:
    # %%
    # Print current experiment info
    log.info(OmegaConf.to_yaml(cfg))

    # Set GPU for current experiment if there's multiple gpu in the environment
    device = T.torch.multiprocessing_device(gpu_id=cfg.gpu_id)
    T.torch.seed(cfg.random.seed, strict=cfg.random.strict)
    log.info(f'device: {device}')

    # Assumes the process runs in a new directory (hydra.cwd==True)
    path_dict = {
        'result': Path('result'),
        'network': Path('network'),
    }
    log.info(f'CWD: {os.getcwd()}')

    # Neural network
    network = hydra.utils.instantiate(cfg.nn)
    network.to(device)

    # %%
    # Load data
    ds_train, ds_val, ds_test = D.get_dataset(cfg)

    # %%
    metrics = {
        'accuracy': hs.E.accuracy_score,
    }

    kwargs = {
        'network': network,
        'train_dataset': ds_train,
        'cfg_train': OmegaConf.to_container(cfg.train['update'], resolve=True),
        'criterion': hydra.utils.instantiate(cfg.train.criterion, dataset=ds_train),
        'network_dir': path_dict['network'],

        'cfg_val': OmegaConf.to_container(cfg.train.validation, resolve=True),
        'val_dataset': ds_val,
        'val_metrics': metrics,
        # 'verbose': True,
        'verbose': False,
        'amp': cfg.amp
    }

    # %%
    trainer = hs.Trainer(**kwargs)

    # %%
    trainer.train()  
    trainer.load_best_model()
    trainer.save()

    # %%
    # Testing
    node.model.to(device)

    # %%
    y_encoder = dataset_test.y_encoder.classes_.tolist()
    result = hs.E.test_node(node, dataset_test, batch_size=cfg.test_batch_size, keep_x=True)
    x, y_true, y_score = result['x'], result['y_true'], result['y_score']
    scorer = E.Scorer(RESULT_DIR=path.result)
    scores = scorer(x=x, y=y_true, y_score=y_score, y_encoder=y_encoder, test_cohort=test_cohort, exp_cfg=cfg, plot=cfg.plot)
    scores = T.unnest_dict(scores)

    # %%
    pp=pprint.PrettyPrinter(indent=2)
    content = pp.pformat(scores)
    log.info(content)
    T.write(content, path.result.join('scores.txt'))
    T.save_pickle(scores, 'scores.p')

    # # %%
# %%
if __name__ == '__main__':
    main()
