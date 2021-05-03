# STAR
Welcome to the official page for the STAR dataset (_Science Topic Analysis on Reddit_), a modular dataset for unsupervised, semi-supervised, weakly supervised and inter-supervised learning.

## Overview

## Class Labels
STAR is a dataset of summaries of scientific discoveries, scraped from posts titles with at least 100 upvotes from the subreddit [r/science](https://www.reddit.com/r/science/). The posts are annotated with 4 coarse and 20 fine-graned topic labels from the 20 most common topic flairs:

- Physical
  - Astronomy
  - Chemistry
  - Earth Science
  - Environment
  - Geology
  - Physics
- Life
  - Animal Science 
  - Biology
  - Cancer
  - Neuroscience
  - Paleontology
- Social 
  - Anthropology 
  - Psychology
  - Social Science
- Applied
  - Computer Science 
  - Engineering
  - Epidemiology
  - Health
  - Medicine
  - Nanoscience

The file [``fine_grained_labels.json``](https://github.com/safranchik/star/blob/main/star/config/fine_grained_labels.json) contains the categorical fine-grained categorical labels for all topics, whereas the file [``coarse_grained_labels.json``](https://github.com/safranchik/star/blob/main/star/config/coarse_grained_labels.json) contains the assicuated coarse-grained labels. This may come in handy if you'll be using this dataset for weak supervision.

## Data Splits
The STAR datasset is split into 18453 labeled and 15528 unlabeled training samples. Training data are unbalanced: the most frequent fine-grained class is __health__ with more than 2400 training ocurrences, whereas the least fequent class is computer science with less than 100 training samples. Additionally, the dataset includes a gold dataset which can be used for validation and a test dataset, both containing 1000 fine-grained balanced samples.

## Loading the Data
The process of loading the data has been made easy! 
1) Import the data loader module called ``starloader``:

``from star import starloader``

2) Load the labeled and unlabeled training data, along balanced validation and test sets:

``` 
labeled_train_data = starloader.load_labeled() 
unlabeled_train_data = starloader.load_unlabeled() 
validation_data = starloader.load_gold()
test_data = starloader.load_test()
```
3) Go build some interesting models!
