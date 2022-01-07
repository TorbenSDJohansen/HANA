# HANA
Code related to the paper HANA: A HAndwritten NAme Database for Offline Handwritten Text Recognition.

To replicate our results, follow the steps in the sections below.
Note the following abbreviations used in the code chunks below:
1. `DATADIR`: This is the directory where you store the data. This never changes.
2. `ROOT`: This is the directory where you save a model and its output. This changes between models.

## Download Database

Include also pretrained models here!

## (Maybe) Clone And Prepare Environment

## Train Neural Networks

**Consider if moving below evaluate**

To train the neural network transcribing only last names, use the command below:
```
python train.py --settings ln --root ROOT --datadir DATADIR
```
This will train a neural network in `ROOT/logs/resnet50-multi-branch/` and log to tensorboard.
To follow training using tensorboard, use the command `tensorboard --port 1234 --logdir ROOT` and visit `localhost:1234`.

To train the network transcribing first and last names:
```
python train.py --settings fln --root ROOT --datadir DATADIR
```

To train the network transcribing first, middle, and last names:
```
python train.py --settings fln --root ROOT --datadir DATADIR
```

## Predict/Evaluate 

To evaluate a trained model on the test set, and obtain predictions on all samples in the test set, first make sure to have a trained model in `ROOT`.
Then use the command below for the network transcribing only last names:
```
python evaluate.py --settings ln --root ROOT --datadir DATADIR
```
This will write two files to `ROOT`:
1. `eval_results.pkl`: Contains accuracy and number of test observations.
2. `preds.csv`: Four-column data frame with (filename, labels, predictions, probability) as columns.

For the network transcribing first and last names:
```
python evaluate.py --settings fln --root ROOT --datadir DATADIR
```

For the network transcribing first, middle, and last names:
```
python evaluate.py --settings fln --root ROOT --datadir DATADIR
```

## Perform Matching

To perform matching, first make sure to have a file with predictions.
Then use the command below:
```
python matching.py --root ROOT --datadir DATADIR
```
This will then write two files to `ROOT`:
1. `eval_results_matched.pkl`: Extends `eval_results.pkl` to include accuracy with matching and number of matches made.
2. `preds_matched.csv`: Extends `preds.csv` to include a column of predictions using matching.

If you provided arguments for `--fn-preds` and `--fn-results` when evaluating/predicting, you must also provide those here.
In that case, you do not need to provide an argument for `--root`.

Note that the dictionaries used to perform matching are in `DATADIR`, which is why it must be specified as an argument.

## Calculate Word Accuracies

To obtain word acccuracy rates and accuracy rates at a specified level of recall, first make sure to have a file with predictions.
Suppose this file was called `path/to/preds.csv`.
Then use the command below:
```
python get_accuracies.py --fn-preds path/to/preds.csv
```
This will print the overall word accuracy and the accuracy at 90% recall.

To use a different level of recall, you can specify it using the command `--recall`.

# TODO

- [ ] Explain Resuming training
- [ ] Transfer learning
- [ ] Explaining settings.py
- [ ] Is recall the right word to use? In paper and in code