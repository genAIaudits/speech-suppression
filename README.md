# Data and code for the paper "Identity-related Speech Suppression in Generative AI Content Moderation"

## Reference information
Grace Proebsting, Oghenefejiro Isaacs Anigboro, Charlie M. Crawford, Danaé Metaxa, and Sorelle A. Friedler. [Identity-related Speech Suppression in Generative AI Content Moderation](https://arxiv.org/abs/2409.13725). ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO), 2025.

### Bibtex
```
@inproceedings{proebsting2025identity,
    title={Identity-related Speech Suppression in Generative {AI} Content Moderation},
    author={Proebsting, Grace and Anigboro, Oghenefejiro Isaacs and Crawford, Charlie M and  Metaxa, Dana{\'e} and Friedler, Sorelle A},
    booktitle={Proceedings of the ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization ({EAAMO})},
    year={2025}
}
```

## SCHEMA

| Column Name | Source   | Explanation |
|---|---|---|
| index       | -   | \[0..543858\] Number of items in dataset     |
| dataset_name   | -    | Name of the dataset where row is from       |
| subset_name     | -    | Subset of original data  |
| data_type    | -    | GenAI or Traditional       |
| text    | -    | Dataset text/value   |
| word_length       | -    | Number of words in text  |
| true_label       | -    |  Overall toxicity label |
| Big_identity       | -    | Semi-column separated list of texts big (grouped) identity  |
| Sub_Identities       | -    | Semi-column separated list of texts sub (in group) identity     |
| has_slur       | -   | If text contains slurs from our slurs list   |
| PG score       | Movies/TV Shows    | PG appropriate boolean |
| PG-13 score       | Movies/TV Shows   | PG-13 appropriate boolean  |
| S     | OpenAI   | Subset of OpenAI data with sexual label  |
| H       | OpenAI    | Subset of OpenAI data with hate label |
| V       | OpenAI    | Subset of OpenAI data with violence label  |
| HR     | OpenAI    | Subset of OpenAI data with harassment label  |
| SH       | OpenAI    | Subset of OpenAI data with self-harm label  |
| S3   | OpenAI   | Subset of OpenAI data with sexual/minors label |
| H2    | OpenAI    | Subset of OpenAI data with hate/threatening label  |
| V2      | OpenAI   | Subset of OpenAI data with violence/graphic label  |
| Hate     | TweetEval    | Subset of TweetEval data with hate label  |
| Offensive     | TweetEval    | Subset of TweetEval data with offensive label  |
| severe_toxicity | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with severe toxicity label |
| obscene | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with sexual explicit label |
| sexual_explicit | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with - label  |
| identity_attack | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with identity attack label  |
| insult | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with insult label  |
| threat | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with threat label  |
| jigsaw_score | ME API - `v1alpha1` | The toxicity score of returned by Jigsaw |
| openAI_flag | ME API - `text-moderation-007` | Boolean value of flag by OpenAI |
| openAI_score | ME API - `text-moderation-007` | The max normalized score from OpenAI |
| anthropic_flag       | ME API - `claude-3-haiku-20240307`   | Boolean of anthropic response  |
| google_score      | ME API - Unknown Model   | The max score from google response |
| llama_guard_flag     | ME API - `llamaguard-2-8b`  | Boolean of llama response  |
| OpenAI_data     | - | Tuple containing date of run and OpenAI model name  |
| Anthropic_data | - | Tuple containing date of run and Anthropic model name |
| OctoAI_data | - | Tuple containing date of run and llama model name |
| Jigsaw_data | - | Tuple containing date of run and jigsaw model name |
| Google_data | - | Tuple containing date of run and google model name |

## Data subsets

### Jigsaw Kaggle

Jigsaw Unintended Bias in Toxicity Classification data with a total of 445,293 unique identity tagged elements. We excluded rows with no identity tags. Rows without values in any of these columns -- male, female, transgender, other_gender, heterosexual, homosexual_gay_or_lesbian, bisexual, other_sexual_orientation, christian, jewish, muslim, hindu, buddhist, atheist, other_religion, black, white, asian, latino, other_race_or_ethnicity, physical_disability, intellectual_or_learning_disability, psychiatric_or_mental_illness, other_disability. Further information about data collection, anonymization, and preparation is discussed in [the kaggle page](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview).
(Compiled data too large to be on github.)

### Jigsaw Bias

Data from Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification paper. This subset of the dataset we used contains 60,560 rows. We extrated rows with phrases containing the identities we focussed on in our research. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://dl.acm.org/doi/pdf/10.1145/3278721.3278729).

### Stormfront

Data (10,944 entries) from the Hate speech dataset from a white supremacist forum. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://aclanthology.org/W18-51.pdf).
Compiled data at [Data/Stormfront/stormfront_data_ME.csv](Data/Stormfront/stormfront_data_ME.csv).

### TweetEval

We used the hate (12,962) and offensive (14,100) subset of the data from the Unified Benchmark and Comparative Evaluation for Tweet Classification paper. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://arxiv.org/pdf/2010.12421).
Compiled data for offensive subset at [Data/Tweets/offensive_twitter_data_with_ME.csv](Data/Tweets/offensive_twitter_data_with_ME.csv),
Compiled data for hate subset at [Data/Tweets/hate_twitter_data_with_ME.csv](Data/Tweets/hate_twitter_data_with_ME.csv).

### OpenAI

Complete data (1,680 entries) from the 'A Holistic Approach to Undesired Content Detection in the Real World' paper. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://arxiv.org/abs/2208.03274).
Compiled data at [Data/Markov/data_with_ME.csv](Data/Markov/data_with_ME.csv).

### Movies

Dataset made for this research.
Compiled data at [Data/Movies/TMDB_with_ME.csv](Data/Movies/TMDB_with_ME.csv).

### TV Synopsis

Dataset made for this research.
Compiled lomg IMDB data at Data/TV Shows/long_IMDB_with_ME.csv,
Compiled medium wiki data at Data/TV Shows/mid_wiki_with_ME.csv,
Compiled short tmdb data at Data/TV Shows/short_TMDB_with_ME.csv.

### FILESTRUCTURE

Dataset (compressed): `speech_suppression_database.7z`

### Running Audit

Run in terminal: `python3 code/run_full_audit.py`
