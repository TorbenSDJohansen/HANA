# -*- coding: utf-8 -*-
"""
@author: tsdj
"""

import torch

from torch.nn.functional import nll_loss

import numpy as np


def sequence_loss(yhat, y):
    k = len(yhat)
    return sum([nll_loss(yhat[i], y[:, i]) for i in range(k)]) / k


def sequence_loss_eval(yhat, y):
    k = len(yhat)
    losses = [nll_loss(yhat[i], y[:, i]).item() for i in range(k)]
    return sum(losses) / k, losses


def sequence_acc(yhat, y):
    accs = []
    status = []

    for i in range(len(yhat)):
        yhat_current = torch.argmax(yhat[i], dim=1) # pylint: disable=E1101
        y_current = y[:, i]

        nb_correct = (y_current.eq(yhat_current)).sum().double()
        acc_current = nb_correct / len(y_current)

        accs.append(acc_current.item())
        status.append(y_current == yhat_current)

    seq_acc = sum(
        [all([status[j][i] for j in range(len(yhat))]) for i in range(len(y))]
        ) / len(y)

    return seq_acc, accs


def split_data(data: dict, batch_size: int):
    ''' Splits data to batches. '''
    images = data['image']
    labels = data['label']

    images_split = torch.split(images, batch_size)
    labels_split = torch.split(labels, batch_size)

    nb_obs = len(labels)

    return tuple(zip(images_split, labels_split)), nb_obs


@torch.no_grad()
def predict_sequence(
        models: dict,
        dataset: torch.utils.data.DataLoader,
        device: torch.device, # pylint: disable=E1101
        retrieve_labels: bool,
        nb_passes: int = 1,
        ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray or list):
    """
    Obtain the predictions of one or more models on a dataset as well as the
    estimated probabilities of the entire sequence (assuming independence, i.e.
    estimated as the product on digit probabilities). Iterates over batches of
    a data loader and allows for device management. If multiple models are
    passed as input, averages the probabilities (BEFORE estimating sequence
    probability).

    Parameters
    ----------
    model : dict
        Dictionary of (model name, neural network) used to predict. If more
        than one (key, value) pair, simple average is used as ensemble.
    data : torch.utils.data.DataLoader
        DataLoader to iterate over for batches.
    device : torch.device
        The device to perform the predictions on (CPU/GPU). Note that the model
        must also be present on that device.
    retrieve_labels : bool
        Whether to retrieve the labels from the DataLoader. Note that these may
        not always exist, hence the option NOT to retrieve labels.
    nb_passes : int
        Number of passes of the observation to iterate over. Only makes sense
        to perform multiple passes when the pipe introduces noise. The default
        is 1.

    Returns
    -------
    preds : np.ndarray
        Array of shape (number of observations, length of sequence) with the
        predictions for each element in each sequence.
    seq_prob : np.ndarray
        Array of shape (number of observations,) with the estimated certainty
        of the specific prediction, calculated as the product of the
        certainties of the predictions of the individual elements of the
        sequence.
    files : np.ndarray
        Array of shape (number of observations,) with the filenames, including
        the full path to the file.
    labels : np.ndarray or empty list
        Array of shape (number of observations,) with labels or, provided
        `retrieve_labels` is False, an empty list.

    """
    assert isinstance(dataset.sampler, torch.utils.data.sampler.SequentialSampler)
    assert isinstance(nb_passes, int) and nb_passes > 0

    for model in models.values():
        assert not model.training

    probabilities = {f'{k}_{i}': [] for k in models.keys() for i in range(nb_passes)}
    files = []
    labels = []

    print(f'Performing predictions by averaging across {len(models)} models and {nb_passes} passes!')

    for current_pass in range(nb_passes):
        print(f'Starting pass {current_pass + 1} of {nb_passes} passes.')
        for i, batch in enumerate(dataset, start=1):

            if i % 10 == 0:
                print(f'Predicting on batch {i} of {len(dataset)}.')

            for model_name, model in models.items():
                yhat_batch = model(batch['image'].to(device).float())
                prob = [torch.nn.functional.softmax(x, dim=1) for x in yhat_batch]
                probabilities[f'{model_name}_{current_pass}'].append(
                    [p.cpu().detach().numpy() for p in prob]
                    )

            if current_pass == 0:
                files.append(batch['fname'])

                if retrieve_labels:
                    labels.append(batch['label'].numpy())

    for model_name_i, probs in probabilities.items():
        if not 'probabilities_flat' in locals():
            probabilities_flat = [np.concatenate(
                [probs[i][j] for i in range(len(probs))]
                ) for j in range(len(probs[0]))]
        else:
            for j in range(len(probs[0])):
                probabilities_flat[j] += np.concatenate(
                    [probs[i][j] for i in range(len(probs))]
         )
    # note probs from prior loop
    for j in range(len(probs[0])):
        probabilities_flat[j] /= len(probabilities)

    preds = np.c_[[np.argmax(p, axis=1) for p in probabilities_flat]].T
    digit_probs = np.c_[
        [np.max(p, axis=1) for p in probabilities_flat]
        ].T

    seq_prob = np.prod(digit_probs, axis=1)

    files = np.concatenate(files)

    if retrieve_labels:
        labels = np.concatenate(labels)

    return preds, seq_prob, files, labels


def eval_sequence(
        targets: np.ndarray,
        preds: np.ndarray,
        seq_prob: np.ndarray,
        threshold: float = None,
        ) -> (float, float or None, float or None):
    """
    Obtain the accuracy of predictions (over the entire sequence, i.e. all
    digits must be correct for a prediction to be counted as correct). Further,
    if a threhold value is provided, the accuracy of the predictions where the
    model is "at least" as certain as the threshold value is provided, as well
    as the associated coverage (i.e. how large a share the predictions above
    the threshold constitute).

    Parameters
    ----------
    targets : np.ndarray
        Array of shape (number of observations, length of sequence) with the
        targets for each element in each sequence. Alternatively, the targets
        may be "cleaned" and in vector format, i.e. with shape
        (number of observations,), provided the same clearning has been applied
        to the predictions.
    preds : np.ndarray
        Array of shape (number of observations, length of sequence) with the
        predictions for each element in each sequence. Alternatively, the
        predictions may be "cleaned" and in vector format, i.e. with shape
        (number of observations,), provided the same clearning has been applied
        to the targets.
    seq_prob : np.ndarray
        Array of shape (number of observations,) with the estimated certainty
        of the specific prediction, calculated as the product of the
        certainties of the predictions of the individual elements of the
        sequence.
    threshold : float, optional
        The threshold value used to calculate the accuracy (and coverage) of
        predictions where the sequence probability is above the value If None,
        this is not calculated. The default is None.

    Returns
    -------
    seq_acc : float
        The sequence accuracy.
    seq_acc_threshold : float or None
        The sequence accuracy for predictions where the sequence probability
        is above the threshold. If the threshold is None, this is not
        calculated and a None value is returned.
    coverage_threshold : float or None
        The coverage of the predictions where the sequence probability
        is above the threshold. If the threshold is None, this is not
        calculated and a None value is returned.

    """
    is_equal = targets == preds

    if len(is_equal.shape) == 1:
        is_equal = is_equal.reshape(-1, 1)

    correct = np.prod(is_equal, axis=1)

    seq_acc = np.mean(correct)

    if threshold is not None:
        seq_acc_threshold = np.mean(correct[seq_prob >= threshold])
        coverage_threshold = np.mean(seq_prob >= threshold)
    else:
        seq_acc_threshold = None
        coverage_threshold = None

    return seq_acc, seq_acc_threshold, coverage_threshold
