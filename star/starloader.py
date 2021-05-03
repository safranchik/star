import json
import logging
import math
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

GLOBAL_SEED = 7
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger("star")

""" Loads the topics-to-ix mapping dictionary """
with open(os.path.join(CURRENT_PATH, "topic_to_ix.json")) as f:
    topic_to_ix = json.load(f)


def load_labeled(num_samples=-1, subtopics: list = None, label_type="hard"):
    """
    Loads the labeled training data.
    :param num_samples: number of samples from the labeled training data to load.
    :param subtopics: list of subtopics to load.
    :param label_type: label type [hard/soft].
    :return: loaded labeled training data in the form of a pandas Dataframe.
    """
    return _load_subset(_read("data/labeled_train_data.csv"), num_samples, subtopics, label_type=label_type)


def load_unlabeled(num_samples=-1):
    """
    Loads the unlabeled training data.
    :param num_samples: number of samples from the unlabeled  training data to load.
    :return: loaded unlabeled training data in the form of a pandas Dataframe.
    """
    data = _read("data/unlabeled_train_data.csv")
    _verify_length(num_samples, len(data))
    return data[:num_samples]


def load_gold(num_samples=-1, subtopics: list = None, label_type="hard"):
    """
    Loads the gold data.
    :param num_samples: number of samples from the gold data to load.
    :param subtopics: list of subtopics to load.
    :param label_type: label type [hard/soft].
    :return: loaded gold data in the form of a pandas Dataframe.
    """

    return _load_subset(_read("data/gold_data.csv"), num_samples, subtopics, balanced=True, label_type=label_type)


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


def _load_subset(data, num_samples: int, subtopics: list = None, balanced: bool = False, label_type: str = "hard"):
    """
    Loads a subset of the data.
    """
    np.random.seed(GLOBAL_SEED)
    _verify_length(num_samples, len(data))

    if num_samples == -1 and subtopics is None:
        set_label_type(data, label_type)
        return data

    if subtopics is None:
        subtopics = topic_to_ix.values()
    else:
        if not isinstance(subtopics, list):
            raise AttributeError("Subtopics must be list of topics.")
        subtopics = [topic_to_ix[st] for st in subtopics]

    if balanced and num_samples != -1 and num_samples % len(subtopics) != 0:
        logger.warning("Number of samples requested ({}) is not a multiple of the number of classes ({}). "
                       "Consider changing this to maintain a balanced dataset.".format(num_samples, len(subtopics)))

    if balanced and num_samples != -1:
        samples_per_topic = math.ceil(num_samples / len(subtopics))
    else:
        samples_per_topic = -1

    data = pd.concat([data[data["topic"] == topic][:samples_per_topic] for topic in subtopics], ignore_index=True)
    set_label_type(data, label_type)

    return shuffle(data)[:num_samples]


def set_label_type(data, label_type):
    if label_type == "soft":
        unique_topics = data["topic"].unique()
        for ix, topic in enumerate(unique_topics):
            data['topic'].replace(topic, ix, inplace=True)

        data['topic'] = np.squeeze(
            np.eye(len(unique_topics), dtype=np.int64)[data["topic"].to_numpy().reshape(-1)]).tolist()
