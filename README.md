# HANA
Code related to the paper HANA: A HAndwritten NAme Database for Offline Handwritten Text Recognition.

- [Download Database](#download-database)
- [Replicate Results](#replicate-results)
- [TODO](#todo)
- [Citing](#citing)

## Download Database

## Replicate Results

To replicate our results, follow the steps in the sections below.
Note the following abbreviations used below:
1. `DATADIR`: This is the directory where you store the HANA database.
2. `ROOT`: This is the directory where you save a model and its output. Each neural network has its own `ROOT`.

### Clone Repository and Prepare Environment

To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/HANA.git
```

Then prepare an environment (here using conda and naming the envionment HANA):
```
conda create -n HANA numpy pandas pillow scikit-learn tensorboard
conda activate HANA
conda install pytorch=1.9 torchvision=0.10 torchaudio cudatoolkit=10.2 -c pytorch
```

### Train Neural Networks

To train the neural network transcribing only last names, use the command below:
```
python train.py --settings ln --root ROOT --datadir DATADIR
```
This will train a neural network in `ROOT/logs/resnet50-multi-branch/` and log to tensorboard.
Five percent of the training data will be used for the purpose of logging validation performance.
To follow training using tensorboard, use the command `tensorboard --port 1234 --logdir ROOT` and visit `localhost:1234`.

To train the network transcribing first and last names:
```
python train.py --settings fln --root ROOT --datadir DATADIR
```

To train the network transcribing first, middle, and last names:
```
python train.py --settings fln --root ROOT --datadir DATADIR
```

### Predict/Evaluate

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

#### Without Training A Model
If you have not trained a model yourself, you can use those provided when you downloaded the database.
This way you can exactly replicate the numbers from the paper.
The pretrained models are located in `DATADIR/pretrained` and are:
1. `fln`: The network transcribing first and last names.
2. `fn`: The network transcribing first, middle, and last names.
3. `ln`: The network transcribing last names.

To use, e.g., the model `ln`, you can either copy/paste the folder `DATADIR/pretrained/ln` to somewhere else and use that new location as `ROOT` or you can specify `ROOT` as `DATADIR/pretrained/ln` directly.

### Perform Matching

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

### Calculate Word Accuracies

To obtain word acccuracy rates and word accuracy rates at a specified level of recall, first make sure to have a file with predictions.
Suppose this file is called `path/to/preds.csv`.
Then use the command below:
```
python get_accuracies.py --fn-preds path/to/preds.csv
```
This will print the overall word accuracy and the accuracy at 90% recall.

If you provide a file with predictions that also includes predictions using matching, the word accuracy when using matching will also be printed.

To use a different level of recall, you can specify it using the command `--recall`.

## TODO

- [ ] How to download database
- [x] How to prepare environment, including which packages are needed
- [ ] Explain Resuming training. Maybe use line 242 in networks/expriment.py to change self.epoch before main loop: self.epochs -= epochs
- [ ] Transfer learning
- [ ] Explaining settings.py
- [ ] Is recall the right word to use? In paper and in code
- [ ] Link to paper
- [ ] Use Torch Hub?
- [ ] debug mode

## Citing

### BibTeX

```bibtex
@article{tsdj2022hana,
  author = {XXX},
  title = {XXX},
  year = {XXX},
  journal = {XXX},
}
```
