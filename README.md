# HANA
Code related to the paper HANA: A HAndwritten NAme Database for Offline Handwritten Text Recognition.

- [Download Database](#download-database)
- [Replicate Results](#replicate-results)
- [Replicate Transfer Learning Results](#replicate-transfer-learning-results)
- [TODO](#todo)
- [Citing](#citing)

## Download Database

The database can be downloaded from [Kaggle](https://www.kaggle.com/sdusimonwittrock/hana-database)

## Replicate Results

To replicate our results, follow the steps in the sections below.
Note the following abbreviations used below:
1. `DATADIR`: This is the directory where you store the HANA database.
2. `ROOT`: This is the directory where you save a model and its output. Each neural network should have its own `ROOT`.

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
To train the neural network transcribing only last names, use the command:
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
python train.py --settings fn --root ROOT --datadir DATADIR
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
python evaluate.py --settings fn --root ROOT --datadir DATADIR
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
Then, use appropriate lexicons to match the predictions to.
For the network transcribing last names, use the command:
```
python matching.py --root ROOT --fn-lex-last DATADIR/last_names.npy
```
This will then write two files to `ROOT`:
1. `eval_results_matched.pkl`: Extends `eval_results.pkl` to include accuracy with matching and number of matches made.
2. `preds_matched.csv`: Extends `preds.csv` to include a column of predictions using matching.

For the network transcribing first and last names:
```
python matching.py --root ROOT --fn-lex-last DATADIR/last_names.npy --fn-lex-first DATADIR/first_names.npy
```

For the network transcribing first, middle, and last names:
```
python matching.py --root ROOT --fn-lex-last DATADIR/last_names.npy --fn-lex-first DATADIR/first_names.npy --fn-lex-middle DATADIR/middle_names.npy
```

**Note**: If you provided arguments for `--fn-preds` and `--fn-results` when evaluating/predicting, you must also provide those here.
In that case, you do not need to provide an argument for `--root`.

### Calculate Word Accuracies
To obtain word acccuracy rates and word accuracy rates at a specified level of coverage, first make sure to have a file with predictions.
Suppose this file is called `path/to/preds.csv`.
Then use the command:
```
python get_accuracies.py --fn-preds path/to/preds.csv
```
This will print the overall word accuracy and the accuracy at 90% coverage.

If you provide a file with predictions that also includes predictions using matching, the word accuracy when using matching will also be printed.

To use a different level of coverage, you can specify it using the command `--coverage`.

## Replicate Transfer Learning Results
Replicating our transfer learning results mostly follow from replicating our other results; make sure to read those sections first.
Suppose you are transfer learning from a model `path/to/tl-model.pt`.

### Train Neural Networks
To train the model on the small sample of the Danish census data, use the command:
```
python train.py --settings ln-danish-census-small-tl --root ROOT --datadir DATADIR --fn-pretrained path/to/tl-model.pt
```
To train the model on the same data but not transfer learning from one of the HANA models, use the command:
```
python train.py --settings ln-danish-census-small --root ROOT --datadir DATADIR
```

To train the models (with and without transfer learning from the HANA database, respectively) on the large sample of the Danish census data, use the commands:
```
python train.py --settings ln-danish-census-large-tl --root ROOT --datadir DATADIR --fn-pretrained path/to/tl-model.pt
python train.py --settings ln-danish-census-large --root ROOT --datadir DATADIR
```

If you have not trained a model to transfer learn from yourself, you can use those provided when you downloaded the database.
That is, you can specify `--fn-pretrained DATADIR/pretrained/ln/logs/resnet50-multi-branch/model_389700.pt`.

### Predict/Evaluate
Evaluating a model and obtaining predictions on the test set of the Danish census data is similar to what has previously been described.
For the transfer learning model on the small sample of the Danish census data, use the command:
```
python evaluate.py --settings ln-danish-census-small-tl --root ROOT --datadir DATADIR
```

For the model without transfer learning from the HANA database on the small sample of the Danish census data, the model transfer learning on the large sample, and the model not transfer learning on the large sample, respectively, use the commands:
```
python evaluate.py --settings ln-danish-census-small --root ROOT --datadir DATADIR
python evaluate.py --settings ln-danish-census-large-tl --root ROOT --datadir DATADIR
python evaluate.py --settings ln-danish-census-large --root ROOT --datadir DATADIR
```

#### Without Training A Model
If you have not trained a model yourself, you can use those provided when you downloaded the database; this is done in the same way as described above.
This way you can exactly replicate the numbers from the paper.
The pretrained models are located in `DATADIR/pretrained` and are:
**TODO**
1. `ln-danish-census-large`
2. `ln-danish-census-large-tl`
3. `ln-danish-census-small`
4. `ln-danish-census-small-tl`

To use, e.g., the model `ln-danish-census-large-tl`, you can either copy/paste the folder `DATADIR/pretrained/ln-danish-census-large-tl` to somewhere else and use that new location as `ROOT` or you can specify `ROOT` as `DATADIR/pretrained/ln-danish-census-large-tl` directly.

### Matching and Calculating Word Accuracies
The remaining steps are identical to what has previously been described:
```
python matching.py --root ROOT --fn-lex-last DATADIR/danish-census/last_names.npy
python get_accuracies.py --fn-preds path/to/preds.csv
```

## TODO
- [ ] How to download database
- [x] How to prepare environment, including which packages are needed
- [x] Resuming training
- [x] Transfer learning
- [x] Is recall the right word to use? In paper and in code
- [ ] Link to paper
- [x] Generalize matching to be able to use different lexicons etc. Rewrite README on matching
- [ ] License: Specifically for RA and ResNet, where code taken elsewhere. Method: Subfolder for each file, add license to that folder (i.e. folder will contain a license and one .py file). Fix imports. Then, in README.md, refer (for each) to 1) folder and 2) repo.
- [x] Reformat US census labels to remove spaces
- [x] Make sure lexicons up to date
- [ ] Where to share model weights? Test if possible to use Dropbox

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
