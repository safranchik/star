import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

GLOBAL_SEED = 7
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

""" Loads the topics-to-ix mapping dictionary """
with open(os.path.join(CURRENT_PATH, "topic_to_ix.json")) as f:
    topic_to_ix = json.load(f)


def load_labeled(num_samples=-1, subtopics: list = None):
    """
    Loads the labeled training data.
    :param num_samples: number of samples from the labeled training data to load.
    :return: loaded labeled training data in the form of a pandas Dataframe.
    """
    data = _read("data/labeled_train_data.csv")
    if num_samples == -1:
        num_samples = len(data)

    return _load_subset(data, num_samples, subtopics)


def load_unlabeled(num_samples=-1):
    """
    Loads the unlabeled training data.
    :param num_samples: number of samples from the unlabeled  training data to load.
    :return: loaded unlabeled training data in the form of a pandas Dataframe.
    """
    data = _read("data/unlabeled_data.csv")
    if num_samples == -1:
        num_samples = len(data)
    _verify_length(num_samples, len(data))
    return data[:num_samples]


def load_gold(num_samples=-1, subtopics: list = None):
    """
    Loads the gold data.
    :param num_samples: number of samples from the gold data to load.
    :param subtopics: list of subtopics to load.
    :return: loaded gold data in the form of a pandas Dataframe.
    """
    data = _read("data/gold_data.csv")
    if num_samples == -1:
        num_samples = len(data)

    return _load_subset(data, num_samples, subtopics, balanced=True)


def load_test():
    """
    :return: loaded test data in the form of a pandas Dataframe.
    """
    return _read("data/test_data.csv")


def _read(file_path):
    """
    Reads a given .csv file from disk.
    :return: loaded data in the form of a pandas Dataframe.
    """
    return pd.read_csv(os.path.join(CURRENT_PATH, file_path))


def _verify_length(num_samples, data_length):
    """
    Ensures that the number of samples queried from the data does not exceed the length of the data itself.
    """
    if num_samples > data_length:
        raise AttributeError("Number of samples requested ({}) exceeds length of dataset ({})."
                             .format(num_samples, data_length))


def _load_subset(data, num_samples: int, subtopics: list = None, balanced=False):
    """
    Loads a subset of the data.
    """
    _verify_length(num_samples, len(data))

    if num_samples == len(data):
        return data

    if subtopics is None:
        subtopics = topic_to_ix.keys()
    else:
        if not isinstance(subtopics, list):
            raise AttributeError("Subtopics must be list of topics.")

    if balanced and num_samples % len(subtopics) != 0:
        warnings.warn("Number of samples requested ({}) is not a multiple of the number of classes ({}). "
                      "Consider changing this to maintain a balanced dataset.".format(num_samples, len(subtopics)))

    subset = pd.DataFrame([], columns=["title", "score", "topic"])
    samples_per_topic = math.ceil(num_samples / len(subtopics))

    for topic in map(topic_to_ix.get, map(lambda x: x.lower(), sorted(subtopics))):
        subset = subset.append(data[data["topic"] == topic][:samples_per_topic], ignore_index=True)

    np.random.seed(GLOBAL_SEED)
    return shuffle(subset)[:num_samples]
