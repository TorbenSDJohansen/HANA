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

from util import MAX_INDIVIDUAL_NAME_LEN, MAX_NB_NAMES


class MatchToStr():
    """
    Matches strings to a set of valid strings - such as names.

    Parameters
    ----------
    potential_strs : set
        The valid strings to be matched to.
    cutoff : float
        The lower threshold to performing matching. Must be in [0, 1]. Default
        is 0.6.
    ignore : set
        Cases of `strs_to_match` to ignore, i.e. not perform matching on.
        Relevant to not match e.g. bad cpd, 0=Mangler, etc. Default is None, in
        which case the empty set is used, i.e. no such cases.

    Returns
    -------
    None.

    """
    def __init__(self, potential_strs: set, cutoff: float = 0.6, ignore: set = None):
        if ignore is None:
            ignore = set()

        assert isinstance(potential_strs, set)
        assert 0 <= cutoff <= 1
        assert isinstance(ignore, set)

        self.potential_strs = potential_strs
        self.cutoff = cutoff
        self.ignore = ignore
        self.fuzzy_map = dict()

    def match(self, strs_to_match: np.ndarray, verbose: bool = True) -> (np.ndarray, int, int, dict):
        """
        Matches strings to a set of valid strings - such as names.

        Does not perform matching for strings in `ignore`. Returns UNMATCHABLE
        if match "similarity" is below `cutoff`.

        Parameters
        ----------
        strs_to_match : np.ndarray
            The strings to match agains `self.potential_strs`.
        verbose : bool
            Whether to print progress.

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
        nb_ignore = 0
        start_time = time.time()

        for i, str_to_match in enumerate(strs_to_match):
            if (i + 1) % 1000 == 0 and verbose:
                running_time = time.time() - start_time
                print(f'Progress: {round(i / nb_to_match * 100, 1)}%. ' +
                      f'Run time: {round(running_time, 1)} seconds. ' +
                      f'Per 1000 predictions: {round(running_time * 1000 / i, 1)} seconds. ' +
                      f'Exact: {nb_exact}. Fuzzy: {nb_fuzzy}. Ignored: {nb_ignore}. ' +
                      f'Number "cached": {len(self.fuzzy_map)}.'
                      )

            if str_to_match in self.ignore:
                nb_ignore += 1
                strs_matched.append(str_to_match)
            elif str_to_match in self.potential_strs:
                nb_exact += 1
                strs_matched.append(str_to_match)
            else:
                nb_fuzzy += 1

                if str_to_match in self.fuzzy_map.keys():
                    str_matched = self.fuzzy_map[str_to_match]
                else:
                    near_matches = difflib.get_close_matches(
                        str_to_match, self.potential_strs, n=1, cutoff=self.cutoff,
                        )
                    if len(near_matches) == 0:
                        str_matched = 'UNMATCHABLE'
                    else:
                        str_matched = near_matches[0]
                    self.fuzzy_map[str_to_match] = str_matched

                strs_matched.append(str_matched)

        strs_matched = np.array(strs_matched)

        return strs_matched, nb_exact, nb_fuzzy, nb_ignore


def split_names(names: np.ndarray, max_nb_middle_names: int):
    split = np.char.split(names)
    last_names = np.array([x[-1] for x in split]).reshape((len(names), 1))
    remaining = [x[:-1] for x in split]
    first_names = np.array([x[0] if len(x) > 0 else '' for x in remaining]).reshape((len(names), 1))
    middle_names = np.empty((len(split), max_nb_middle_names), dtype=f'U{MAX_INDIVIDUAL_NAME_LEN}')

    for i, names in enumerate([x[1:] for x in remaining]):
        for j, name in enumerate(names):
            middle_names[i, j] = name

    names = np.concatenate([first_names, middle_names, last_names], axis=1)

    return names


def match(names: np.ndarray, lookup: dict):
    nb_cols = names.shape[1]
    names_matched = names.copy()

    middle_name_matcher = MatchToStr(lookup['middle']) # for re-use
    mapper = {
        0: MatchToStr(lookup['first']),
        **{i: middle_name_matcher for i in range(1, nb_cols - 1)},
        (nb_cols - 1): MatchToStr(lookup['last']),
        }
    nb_fuzzy = {}

    for i, matcher in mapper.items():
        print(f'Matching column {i + 1} of {nb_cols}.')
        verbose = len(matcher.potential_strs) > 1
        matched, _, nb_fuzzy_i, _ = matcher.match(names[:, i], verbose)
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

    parser.add_argument('--fn-lex-first', type=str, default=None)
    parser.add_argument('--fn-lex-middle', type=str, default=None)
    parser.add_argument('--fn-lex-last', type=str, default=None)

    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--fn-results', type=str, default=None)
    parser.add_argument('--fn-preds', type=str, default=None)

    parser.add_argument('--max-nb-middle-names', type=int, default=MAX_NB_NAMES - 2)
    parser.add_argument('--allow-empty', type=str, nargs='+', default=['first', 'middle'])

    args = parser.parse_args()

    return args


def construct_lookup(args):
    first = pd.read_csv(
        args.fn_lex_first, keep_default_na=False,
        ).values.reshape(-1) if args.fn_lex_first else ['']
    middle = pd.read_csv(
        args.fn_lex_middle, keep_default_na=False,
        ).values.reshape(-1) if args.fn_lex_middle else ['']
    last = pd.read_csv(
        args.fn_lex_last, keep_default_na=False,
        ).values.reshape(-1) if args.fn_lex_last else ['']

    lookup = {'first': set(first), 'middle': set(middle), 'last': set(last)}

    for allow_empty in args.allow_empty:
        lookup[allow_empty] = lookup[allow_empty].union(set(['']))

    return lookup


def main():
    args = parse_args()

    if args.root is None:
        assert args.fn_results is not None and args.fn_preds is not None

    fn_results = args.fn_results
    fn_preds = args.fn_preds

    if fn_results is None:
        fn_results = os.path.join(args.root, 'eval_results.pkl')
    if fn_preds is None:
        fn_preds = os.path.join(args.root, 'preds.csv')

    preds = pd.read_csv(fn_preds)
    results = pickle.load(open(fn_results, 'rb'))

    lookup = construct_lookup(args)
    split_preds = split_names(preds['pred'].values.astype('U'), args.max_nb_middle_names)
    matched, nb_matches = match(split_preds, lookup)

    preds['pred_m'] = matched
    acc = (preds['pred_m'] == preds['label']).mean()

    results.update(nb_matches)
    results['Full sequence accuracy (with matching)'] = acc
    print(results)

    pickle.dump(results, open(fn_results.replace('.pkl', '_matched.pkl'), 'wb'))
    preds.to_csv(fn_preds.replace('.csv', '_matched.csv'), index=False)


if __name__ == '__main__':
    main()
