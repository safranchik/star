# STAR
Welcome to the official page for the STAR dataset (_Science Topic Analysis on Reddit_), a modular dataset for unsupervised, semi-supervised, weakly supervised and inter-supervised learning.

## Overview

STAR is a dataset of summaries of scientific discoveries, scraped from the titles of posts from the subreddit [r/science](https://www.reddit.com/r/science/) with at least 100 upvotes. We labeled flaired posts with their corresponding topic flair, and used unflaired posts as unlabeled data. The 20 topic labels are the following:

- Animal Science
- Anthropology
- Astronomy
- Biology
- Cancer
- Chemistry
- Computer Science
- Earth Science
- Engineering
- Environment
- Epidemiology
- Geology
- Health
- Medicine
- Nanoscience
- Neuroscience
- Paleontology
- Physics
- Psychology
- Social Science


Due to the lack of samples, we discarded the Mathematics topic, and merged Economics into Social Science. The file [topic_to_ix.json](https://github.com/safranchik/STAR/blob/main/data/topic_to_ix.json) contains the categorical labels for all topics. This may come in handy if you'll be using this dataset for weak supervision.

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
