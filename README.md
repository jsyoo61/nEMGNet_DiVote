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

2. Classifier training
```
python train2.py
```

# Analysis
Tbd