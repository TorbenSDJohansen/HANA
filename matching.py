# -*- coding: utf-8 -*-
"""
@author: sa-tsdj

Helper functions to prepare and present results.

"""

import time
import difflib

import numpy as np

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
