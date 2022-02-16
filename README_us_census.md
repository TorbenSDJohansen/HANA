## Replicate US Census Results

### Evaluation

<details>

<summary>
(large subset) us-census model on us-census-last-name
</summary>

```
python evaluate.py --settings US-census-large-last-name --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/t5bvr6oh27p4wcs/us-census-large-last-name.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

<details>

<summary>
(large subset) us-census model w/ TL on us-census-last-name
</summary>

```
python evaluate.py --settings US-census-large-last-name-tl --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/mb73ce9wgqf4er6/us-census-large-last-name-tl.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

<details>

<summary>
(small subset) us-census model on us-census-last-name
</summary>

```
python evaluate.py --settings US-census-small-last-name --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/2u4nfrkb0wof017/us-census-small-last-name.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

<details>

<summary>
(small subset) us-census model w/ TL on us-census-last-name
</summary>

```
python evaluate.py --settings US-census-small-last-name-tl --root ROOT --datadir DATADIR --model-from-url https://www.dropbox.com/s/nvtmvih13ttac9a/us-census-small-last-name-tl.pt?dl=1
python matching.py --root ROOT --fn-lex-last DATADIR/labels/lexicon/last_names.csv
python get_accuracies.py --fn-preds ROOT/preds_matched.csv
```

</details>

### Training and transfer learning

<details>

<summary>
(large subset) us-census model on us-census-last-name
</summary>

```
python train.py --settings US-census-large-last-name --root ROOT --datadir DATADIR
```

</details>

<details>

<summary>
(large subset) us-census model on us-census-last-name w/ TL
</summary>

```
python train.py --settings US-census-large-last-name-tl --root ROOT --datadir DATADIR --url-pretrained https://www.dropbox.com/s/vwba88pta7qc2qr/hana-last-name.pt?dl=1
```

</details>

<details>

<summary>
(small subset) us-census model on us-census-last-name
</summary>

```
python train.py --settings US-census-small-last-name --root ROOT --datadir DATADIR
```

</details>

<details>

<summary>
(small subset) us-census model on us-census-last-name w/ TL
</summary>

```
python train.py --settings US-census-small-last-name-tl --root ROOT --datadir DATADIR --url-pretrained https://www.dropbox.com/s/vwba88pta7qc2qr/hana-last-name.pt?dl=1
```

</details>