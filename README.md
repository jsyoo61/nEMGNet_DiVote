# nEMGNet_DiVote
Code for ([Yoo, Jaesung, et al.](https://doi.org/10.1016/j.cmpb.2022.107079) "Residual one-dimensional convolutional neural network for neuromuscular disorder classification from needle electromyography signals with explainability." Computer Methods and Programs in Biomedicine 226 (2022): 107079.)

Powered by [hydra](https://hydra.cc/docs/intro/), [hideandseek](https://github.com/jsyoo61/hideandseek), and [tools-jsyoo61](https://github.com/jsyoo61/tools)

# Preprocess
Recommend generating toy data to check format
```
python preprocess.py toy=true
```

To use your own data, put the data in `data/raw/` directory.
Data should be:
- Signal files
- `cohort.csv` indicating labels for each patient
- `cohort_signal.csv` indicating labels for each signal

# Training
Training is done in 2 steps:

1. nEMGNet training

```
python train1.py
```

To specify specific run directory:
```
python train1.py hydra.run.dir=your_wanted_dir
```
For a sweep run
```
python train1.py -m "random.seed=range(0,5)" train.update.lr=1e-3,1e-4 train.update.batch_size=64,128 hydra.sweep.dir=your_sweep_dir
```

2. Classifier training

Use patient feature without muscle type info
```
python train2.py feature_type=all dir.train1=your_wanted_dir
```
Use patient feature with muscle type info
```
python train2.py feature_type=PD dir.train1=your_wanted_dir
```

Sweep over train1 sweep directories
```
python train2.py feature_type=all "dir.train1=your_wanted_dir/${subdir}" "subdir=range(0,20)"
```

# Analysis
Tbd