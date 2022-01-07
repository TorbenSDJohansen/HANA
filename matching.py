# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import os
import time
import difflib
import argparse
import pickle
import re

import pandas as pd
import numpy as np

from util import MAX_INDIVIDUAL_NAME_LEN, MAX_NB_MIDDLE_NAMES


class MatchToStr():
    """
    Matches strings to a set of valid strings - such as names.

    Parameters
    ----------
    potential_strs : set
        The valid strings to be matched to.

    Returns
    -------
    None.

    """
    def __init__(self, potential_strs: set):
        assert isinstance(potential_strs, set)
        self.potential_strs = potential_strs
        self.fuzzy_map = dict()

    def match(self, strs_to_match: np.ndarray) -> (np.ndarray, int, int, dict):
        """
        Matches strings to a set of valid strings - such as names.

        Parameters
        ----------
        strs_to_match : np.ndarray
            The strings to match agains `self.potential_strs`.

        Returns
        -------
        strs_matched : np.ndarray
            The modified input, where all strings not in `self.potential_strs`
            have been matched to the nearest valid string.
        nb_exact : int
            The number of exact matches perfored, i.e. where the string already
            existed in `self.potential_strs`.
        nb_fuzzy : int
            Number of times where it was needed to match to nearest valid.
        self.fuzzy_map : dict
            Dictionary where the keys are the strings that were not valid and
            the values the strings they were matched to.

        """
        nb_to_match = len(strs_to_match)
        strs_matched = []
        nb_exact = 0
        nb_fuzzy = 0
        start_time = time.time()

        for i, str_to_match in enumerate(strs_to_match):
            if (i + 1) % 1000 == 0:
                running_time = time.time() - start_time
                print(f'Progress: {round(i / nb_to_match * 100, 1)}%. ' +
                      f'Run time: {round(running_time, 1)} seconds. ' +
                      f'Per 1000 predictions: {round(running_time * 1000 / i, 1)} seconds. ' +
                      f'Exact: {nb_exact}. Fuzzy: {nb_fuzzy}. ' +
                      f'Number "cached": {len(self.fuzzy_map)}.')

            if str_to_match in self.potential_strs:
                nb_exact += 1
                strs_matched.append(str_to_match)
            else:
                nb_fuzzy += 1

                if str_to_match in self.fuzzy_map.keys():
                    str_matched = self.fuzzy_map[str_to_match]
                else:
                    near_matches = difflib.get_close_matches(str_to_match, self.potential_strs)
                    str_matched = near_matches[0]
                    self.fuzzy_map[str_to_match] = str_matched

                strs_matched.append(str_matched)

        strs_matched = np.array(strs_matched)

        return strs_matched, nb_exact, nb_fuzzy, self.fuzzy_map


def split_names(names: np.ndarray):
    split = np.char.split(names)
    last_names = np.array([x[-1] for x in split]).reshape((len(names), 1))
    remaining = [x[:-1] for x in split]
    first_names = np.array([x[0] if len(x) > 0 else '' for x in remaining]).reshape((len(names), 1))
    middle_names = np.empty((len(split), MAX_NB_MIDDLE_NAMES), dtype=f'U{MAX_INDIVIDUAL_NAME_LEN}')

    for i, names in enumerate([x[1:] for x in remaining]):
        for j, name in enumerate(names):
            middle_names[i, j] = name

    names = np.concatenate([first_names, middle_names, last_names], axis=1)

    return names


def match(names: np.ndarray, lookup: dict):
    nb_cols = names.shape[1]
    names_matched = names.copy()

    middle_name_matcher = MatchToStr(lookup['middle'].union(set(['']))) # for re-use
    mapper = {
        0: MatchToStr(lookup['first'].union(set(['']))),
        **{i: middle_name_matcher for i in range(1, nb_cols - 1)},
        (nb_cols - 1): MatchToStr(lookup['last']),
        }
    nb_fuzzy = {}

    for i, matcher in mapper.items():
        print(f'Matching column {i + 1} of {nb_cols}.')
        matched, _, nb_fuzzy_i, _ = matcher.match(names[:, i])
        names_matched[:, i] = matched
        nb_fuzzy[i] = nb_fuzzy_i

    nb_matches = {
        'Number of first names matched': nb_fuzzy[0],
        'Number of middle names matched': sum(nb_fuzzy[i] for i in range(1, nb_cols - 1)),
        'Number of last names matched': nb_fuzzy[nb_cols - 1],
        }

    names_matched = pd.DataFrame(names_matched)
    names_matched_flat = names_matched[0]

    for i in range(1, nb_cols):
        names_matched_flat += ' ' + names_matched[i]

    names_matched_flat = names_matched_flat.apply(lambda x: re.sub(' +', ' ', x).strip())

    return names_matched_flat.values, nb_matches


def parse_args():
    parser = argparse.ArgumentParser(description='Matching')

    parser.add_argument('--root', type=str)
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--fn-results', type=str, default=None)
    parser.add_argument('--fn-preds', type=str, default=None)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    fn_results = args.fn_results
    fn_preds = args.fn_preds

    if fn_results is None:
        fn_results = os.path.join(args.root, 'eval_results.pkl')
    if fn_preds is None:
        fn_preds = os.path.join(args.root, 'preds.csv')

    preds = pd.read_csv(fn_preds)
    results = pickle.load(open(fn_results, 'rb'))

    lookup_str = os.path.join(args.datadir, '{}_names.npy')
    lookup = {
        'first': set(np.load(lookup_str.format('first'), allow_pickle=True)),
        'middle': set(np.load(lookup_str.format('middle'), allow_pickle=True)),
        'last': set(np.load(lookup_str.format('last'), allow_pickle=True)),
        }

    split_preds = split_names(preds['pred'].values.astype('U'))
    matched, nb_matches = match(split_preds, lookup)

    preds['pred_m'] = matched
    acc = (preds['pred_m'] == preds['label']).mean()

    results.update(nb_matches)
    results['Accuracy (with matching)'] = acc

    pickle.dump(results, open(fn_results.replace('.pkl', '_matched.pkl'), 'wb'))
    preds.to_csv(fn_preds.replace('.csv', '_matched.csv'), index=False)


if __name__ == '__main__':
    main()
