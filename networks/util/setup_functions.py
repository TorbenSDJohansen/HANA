# -*- coding: utf-8 -*-
"""
@author: tsdj

Script to contain a few simple functions to avoid code copying between scripts.
"""

import os
import json

import numpy as np

def _remove_labels_with_no_image(image_dir: str, labels: np.ndarray) -> np.ndarray:
    ''' This method ensures that only files where we have the label and
    the image are maintained.
    '''
    image_files = os.listdir(image_dir)

    labels_dict = {labels[i, 0]: labels[i, 1] for i in range(len(labels))}
    overlap = set(image_files).intersection(set(labels[:, 0]))
    labels_dict_pruned = {
        k: v for k, v in labels_dict.items() if k in overlap
        }

    labels_array = np.array(
        [[x, y] for x, y in labels_dict_pruned.items()],
        dtype=object,
        )

    return labels_array


def prepare_labels(cells: list, root_labels: str, root_images: str) -> np.ndarray:
    """
    Convinience for label load. Takes a list of cells (the naming is used to
    refer to the fact that we often use minipics of cells from tables - they
    need not be cells. Just any list of strings such that the strings are
    names of the label files in `root_labels` and subfolders in `root_images`).
    The function works by considering each element of `cells` and "mapping" to
    the labels and images by using, respectively, `root_labels` and
    `root_images`.
    Then, one element at a time, it loads the labels and ensures that only
    labels where the associated image exists are kept. This solves problems
    where our labels contain images we have not been able to crop, for example.
    It also adds the path to the filename (such that a simple filename is
    transformed to a full path + filename).
    Finally, all label files are concatenated to create one array of labels.
    Parameters
    ----------
    cells : list
        List of strings, each of which refers to a label file (an .npy file -
        it MUST be an .npy file, the first column of which is the filename and
        the second column is the label) and a folder with images.
        The elements may alternatively be tuples of exactly 2 elements, the
        first of which refers to the label file and the second of which refers
        to the image directory. This is useful if the name of the label file
        and its corrosponding image directory does not match.
        You may mix strings and 2-tuples.
    root_labels : str
        The directory containing all the label files (.npy files).
    root_images : str
        The directory containing all the image folders. Images can be any
        standard type (.png, .jpg, etc.).
    Returns
    -------
    labels_merged : np.ndarray
        Array of labels. Consist of two columns, this first of which is the
        filename (WITH full path) and the second of which is the labels.

    """
    cells = cells.copy()

    for i, element in enumerate(cells):
        if isinstance(element, str):
            cells[i] = (element, element)
        else:
            assert isinstance(element, tuple) and len(element) == 2

    labels_info = {cell: {
        'labels': ''.join((root_labels, cell[0], '.npy')),
        'image_dir': ''.join((root_images, cell[1], '/')),
        } for cell in cells}
    for value in labels_info.values():
        value['array'] = _remove_labels_with_no_image(
            image_dir=value['image_dir'],
            labels=np.load(value['labels'], allow_pickle=True),
            )
        value['array'][:, 0] = [
            ''.join((value['image_dir'], f)) for f in value['array'][:, 0]
            ] # add path
    labels_merged = np.concatenate(
        [value['array'] for value in labels_info.values()]
        )
    return labels_merged


def prepare_fnames(cells: list, root_images: str, filetypes: str or tuple = None) -> np.ndarray:
    """
    Prepares an array of image file names to be used for, for example,
    prediction from models.

    Parameters
    ----------
    cells : list
        List of strings, each of which refers to a a folder with images. In
        case the list consists of 2-tuples (possible in `prepare_labels`), the
        second element of the tuples are correctly used to identify the image
        directory.
    root_images : str
        The directory containing all the image folders. Images can be any
        standard type (.png, .jpg, etc.).
    filetypes : str or tuple, optional
        On optional option to filter files from a folder. Only file types
        matching those specified by `filetypes` are kept. The default is None.

    Returns
    -------
    image_files_array : np.ndarray
        Array of shape (n,) with filesnames. Importantly, these include the
        full path to the file.

    """
    cells = cells.copy()

    assert isinstance(cells, (list, tuple))
    assert len(set(cells)) == len(cells)

    for i, element in enumerate(cells):
        if isinstance(element, tuple):
            assert len(element) == 2
            cells[i] = element[1]
        else:
            assert isinstance(element, str)

    image_dirs = {c: ''.join((root_images, c, '/')) for c in cells}
    image_files = {c: os.listdir(imdir) for c, imdir in image_dirs.items()}

    if filetypes is not None:
        if isinstance(filetypes, str):
            filetypes = (filetypes,)
        else:
            assert isinstance(filetypes, tuple)

        for key, value in image_files.items():
            image_files[key] = [x for x in value if x.split('.')[-1] in filetypes]

    for key, value in image_files.items():
        image_files[key] = [''.join((image_dirs[key], x)) for x in value]

    image_files_array = np.concatenate(list(image_files.values()))

    # Below check should be redundant... maybe remove it
    assert len(set(image_files_array)) == len(image_files_array)

    return image_files_array


def get_model_file(root: str, model_name: str):
    ''' Uses a root folder and a model name to retrieve the newest model, i.e.
    the one trained for the highest number of steps.
    '''
    directory = f'{root}/logs/{model_name}'
    meta_info = json.load(open(f'{directory}/meta_info.json', 'r'))
    model_file = f'{directory}/model_{meta_info["step"]}.pt'

    return model_file
