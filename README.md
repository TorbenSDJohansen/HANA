# HANA
This repository contains the code related to the paper [HANA: A HAndwritten NAme Database for Offline Handwritten Text Recognition](https://www.sciencedirect.com/science/article/pii/S0014498322000511) by Christian M. Dahl, Torben Johansen, Emil N. Sørensen, and Simon Wittrock (earlier [arXiv version](https://arxiv.org/abs/2101.10862)).

- [Download Database](#download-database)
- [Clone Repository and Prepare Environment](#clone-repository-and-prepare-environment)
- [Replicate Results](#replicate-results)
- [License](#license)
- [Citing](#citing)

## Download Database

### HANA database
The HANA database can be downloaded from [Kaggle](https://www.kaggle.com/sdusimonwittrock/hana-database).
The images are from the police register sheets from Copenhagen which cover all adults (above the age of 10) residing in the capital of Denmark, Copenhagen, in the period from 1890 to 1923.
The labels are names written in lower case letters and contain only characters which are used in Danish words, which implies 29 alphabetic characters, i.e. this database includes the letters æ, ø, and å.

### Other databases
Aside from the HANA database, Danish and US census data is used to conduct two of our transfer learning illustrations.
Both of these come in a small and large version.
The small sample Danish census data can be downloaded from [Kaggle](https://www.kaggle.com/sdusimonwittrock/danish-census-handwritten-names-small).
The large sample Danish census data can be downloaded from [Kaggle](https://www.kaggle.com/sdusimonwittrock/danish-census-handwritten-names-large).
The US census data we use is not publicly available currently.

## Clone Repository and Prepare Environment

To get started, first clone the repository locally:
```
git clone https://github.com/TorbenSDJohansen/HANA.git
```

Then prepare an environment (here using conda and the name HANA):
```
conda create -n HANA numpy pandas pillow scikit-learn tensorboard setuptools=58
conda activate HANA
conda install pytorch=1.9 torchvision=0.10 torchaudio cudatoolkit=10.2 -c pytorch
```

### Model Zoo
Our trained models are available for download:

<details>

<summary>
Table with HANA models and URLs
</summary>

| name                      | WACC  | WACC w/ matching  | url |
| ---                       | ---   | ---               | --- |
| hana-last-name            | 94.3  | 95.7              | [model](https://www.dropbox.com/s/vwba88pta7qc2qr/hana-last-name.pt?dl=1)           |
| hana-first-and-last-name  | 93.5  | 94.8              | [model](https://www.dropbox.com/s/1zbfd7l3bkdg662/hana-first-and-last-name.pt?dl=1) |
| hana-full-name            | 67.4  | 68.8              | [model](https://www.dropbox.com/s/jj32kp5sy6bdmoh/hana-full-name.pt?dl=1)           |

</details>

<details>

<summary>
Table with Danish census models and URLs
</summary>

| name                      | WACC  | url |
| ---                       | ---   | --- |
| danish-census-large       | 86.1  | [model](https://www.dropbox.com/s/bcobjqiolvcdte6/danish-census-large-last-name.pt?dl=1)    |
| danish-census-large-tl    | 94.6  | [model](https://www.dropbox.com/s/rbd6ibyrnjqycgs/danish-census-large-last-name-tl.pt?dl=1) |
| danish-census-small       | 77.8  | [model](https://www.dropbox.com/s/i2jjk905vrcc4op/danish-census-small-last-name.pt?dl=1)    |
| danish-census-small-tl    | 92.2  | [model](https://www.dropbox.com/s/2v8g1lb0rhrjx6z/danish-census-small-last-name-tl.pt?dl=1) |

</details>

<details>

<summary>
Table with US census models and URLs
</summary>

| name                      | WACC  | url |
| ---                       | ---   | --- |
| us-census-large           | 84.7  | [model](https://www.dropbox.com/s/t5bvr6oh27p4wcs/us-census-large-last-name.pt?dl=1)    |
| us-census-large-tl        | 86.8  | [model](https://www.dropbox.com/s/mb73ce9wgqf4er6/us-census-large-last-name-tl.pt?dl=1) |
| us-census-small           | 72.8  | [model](https://www.dropbox.com/s/2u4nfrkb0wof017/us-census-small-last-name.pt?dl=1)    |
| us-census-small-tl        | 78.7  | [model](https://www.dropbox.com/s/nvtmvih13ttac9a/us-census-small-last-name-tl.pt?dl=1) |

</details>

## Replicate Results

To replicate our results, follow the steps in the following sections.
Note the following abbreviations used:
1. `DATADIR`: This is the directory where you store the data (images, labels, lexicons). This varies between datasets, i.e. the `DATADIR` for the HANA database is different from the one for the Danish census, which itself comes in both a small and a large version.
2. `ROOT`: This is the directory where you store a model and its output (such as predictions). Each neural network should have its own `ROOT`.

While the US census data we use is not publicly available, the commands we use are still available: [README_us_census.md](README_us_census.md).

### Evaluation

To evaluate the pre-trained hana-last-name model on the hana-last-name data (having downloaded the HANA database to the folder `DATADIR` and storing the model's output in the folder `ROOT`):
```
python evaluate.py --settings hana-last-name --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/vwba88pta7qc2qr/hana-last-name.pt?dl=1
```
This will print the full sequence accuracy and write two files to `ROOT`:
1. `eval_results.pkl`: Contains accuracy and number of test observations.
2. `preds.csv`: Four-column data frame with (filename, labels, predictions, probability) as columns.

To further use matching:
```
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
```
This will print the full sequence accuracy (also with matching) and write two files to `ROOT`:
1. `eval_results_matched.pkl`: Extends `eval_results.pkl` to include accuracy with matching and number of matches made.
2. `preds_matched.csv`: Extends `preds.csv` to include a column of predictions using matching.

To obtain word acccuracy rates and word accuracy rates at a specified level of coverage:
```
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```
This will print the overall word accuracy and the word accuracy at 90% coverage.

#### Remaining models on HANA database
<details>

<summary>
hana-first-and-last-name model on hana-first-and-last-name
</summary>

```
python evaluate.py --settings hana-first-and-last-name --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/1zbfd7l3bkdg662/hana-first-and-last-name.pt?dl=1
python matching.py --root ROOT --fn-lex-first DATADIR/labels/lexicon/first_names.csv --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

<details>

<summary>
hana-full-name model on hana-full-name
</summary>

```
python evaluate.py --settings hana-full-name --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/jj32kp5sy6bdmoh/hana-full-name.pt?dl=1
python matching.py --root ROOT --fn-lex-first DATADIR/labels/lexicon/first_names.csv --fn-lex-middle DATADIR/labels/lexicon/middle_names.csv --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

#### Danish census
<details>

<summary>
(large subset) danish-census model on danish-census-last-name
</summary>

```
python evaluate.py --settings danish-census-large-last-name --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/bcobjqiolvcdte6/danish-census-large-last-name.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

<details>

<summary>
(large subset) danish-census model w/ TL on danish-census-last-name
</summary>

```
python evaluate.py --settings danish-census-large-last-name-tl --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/rbd6ibyrnjqycgs/danish-census-large-last-name-tl.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

<details>

<summary>
(small subset) danish-census model on danish-census-last-name
</summary>

```
python evaluate.py --settings danish-census-small-last-name --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/i2jjk905vrcc4op/danish-census-small-last-name.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

<details>

<summary>
(small subset) danish-census model w/ TL on danish-census-last-name
</summary>

```
python evaluate.py --settings danish-census-small-last-name-tl --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/2v8g1lb0rhrjx6z/danish-census-small-last-name-tl.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>



### Training and transfer learning
To train the model on the hana-last-name data (having downloaded the HANA database to the folder `DATADIR` and storing the model and logs in the folder `ROOT`):
```
python train.py --settings hana-last-name --root ROOT --datadir DATADIR
```
This will train a neural network in `ROOT/logs/resnet50-multi-branch/` and log to tensorboard.
Five percent of the training data will be used for the purpose of logging validation performance.
To follow training using tensorboard, use the command `tensorboard --port 1234 --logdir ROOT` and visit `localhost:1234`.

<details>

<summary>
hana-first-and-last-name model
</summary>

```
python train.py --settings hana-first-and-last-name --root ROOT --datadir DATADIR
```

</details>

<details>

<summary>
hana-full-name model
</summary>

```
python train.py --settings hana-full-name --root ROOT --datadir DATADIR
```

</details>

#### Danish census
<details>

<summary>
(large subset) danish-census model
</summary>

```
python train.py --settings danish-census-large-last-name --root ROOT --datadir DATADIR
```

</details>

<details>

<summary>
(large subset) danish-census model w/ TL
</summary>

```
python train.py --settings danish-census-large-last-name-tl --root ROOT --datadir DATADIR --url-pretrained https://www.dropbox.com/s/vwba88pta7qc2qr/hana-last-name.pt?dl=1
```

</details>

<details>

<summary>
(small subset) danish-census model
</summary>

```
python train.py --settings danish-census-small-last-name --root ROOT --datadir DATADIR
```

</details>

<details>

<summary>
(small subset) danish-census model w/ TL
</summary>

```
python train.py --settings danish-census-small-last-name-tl --root ROOT --datadir DATADIR --url-pretrained https://www.dropbox.com/s/vwba88pta7qc2qr/hana-last-name.pt?dl=1
```

</details>

## License

Our code is licensed under MIT (see [LICENSE](LICENSE)).
Part of the code used originates from two other libraries.
To make explicit what originates from elsewhere, those parts are located in their own directories (in a modified form and only the part used in this project).
In those directories, the license under which the library is licensed is included.
1. **RandAugment**: Originates from [pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment) and located in [networks/augment/pytorch_randaugment](networks/augment/pytorch_randaugment).
1. **ResNet**: Originates from [torchvision](https://github.com/pytorch/vision) and located in [networks/util/torchvision](networks/util/torchvision).

## Citing
If you would like to cite our work, please use
```bibtex
@article{dahl2023hana,
    author = {Dahl, Christian M. and Johansen, Torben S. D. and S{\o}rensen, Emil N. and Wittrock, Simon F.},
    title = {HANA: A HAndwritten NAme Database for Offline Handwritten Text},
    journal = {Explorations in Economic History},
    volume = {87},
    pages = {101473},
    year = {2023},
    note = {Methodological Advances in the Extraction and Analysis of Historical Data},
    issn = {0014-4983},
    doi = {https://doi.org/10.1016/j.eeh.2022.101473},
    url = {https://www.sciencedirect.com/science/article/pii/S0014498322000511}
}
```
or (as a reference to the earlier arXiv version)
```bibtex
@misc{dahl2022hana,
      title={HANA: A HAndwritten NAme Database for Offline Handwritten Text Recognition}, 
      author={Christian M. Dahl and Torben Johansen and Emil N. Sørensen and Simon Wittrock},
      year={2022},
      eprint={2101.10862},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
