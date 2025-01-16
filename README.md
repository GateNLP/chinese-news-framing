# Chinese News Framing
Repository for Chinese News Framing Dataset to be integrated with the SemEval dataset, or used as a standalone dataset,
potentially useful for classification methods utilising disagreement at the annotator or annotation level.

## Code

Uses the same structure as the code from [https://doi.org/10.48550/arXiv.2410.14515](https://doi.org/10.48550/arXiv.2410.14515).

## Instructions

### Getting Started
First, clone this repository with
```
git clone git@github.com:GateNLP/chinese-news-framing.git
```

Once the repository has been cloned, the dataset will need to be downloaded
from Zenodo and saved to `data/raw/chinese_news_framing_dataset.csv` from the base of this repository.

Zenodo DOI: [https://doi.org/10.5281/zenodo.14659362](https://doi.org/10.5281/zenodo.14659362)


### Add Text to the Dataset
Due to copyright issues, the text could not be released in the published
version of the Chinese News Framing Dataset. For this reason, we add
the `src/dataset_creation/add_text.py` script, allowing for the scraping
of the news content (title and main content).

This script is provided for academic and research purposes only. Users are
responsible for ensuring compliance with local laws and the terms of service
of the websites they scrape. The authors of this script do not condone or 
encourage the use of this tool for any unlawful or unethical activities.

Add the text by running this script, which adds the dataset to
`data/chinese_news_framing_with_text.csv`:
```
python src/dataset_creation/add_text.py
```

Once this text has been loaded, perform the text preprocessing steps
outlined in the paper:
* a newline character is added between the news title and body;
* duplicate sentences occurring consecutively are removed;
* hyperlinks to websites and images are removed;
* strings detailing author biographies (such as names and affiliations) are also removed.

This should be performed over the whole SemEval dataset too.


### Split the Data into Train, Dev, and Test Sets
The `src/dataset_creation/split_data.py` script has been created to split
the dataset into three separate datasets: train, dev, and test. These datasets
can then be used directly in the pipeline in `src/sem_main.py`. The
name of the dataset may need to be updated depending on whether you would
like to further merge the data with the SemEval set like we did. Our
configuration files use the names `train_set.csv`, `dev_set.csv`, `test_set.csv`.

Run this script with:
```
python src/dataset_creation/split_data.py
```

Not that if you do not merge this dataset with the SemEval set, it will
not be possible to use the test settings `sem_only` and `all`; only
`chinese_only` will be available.


### Combine with SemEval Dataset
Once the train, dev, and test splits have been created using the cleaned text data,
concatenate the SemEval training set with the `train_set.csv` and save to
`data/semeval_train_clean.csv`. Then concatenate the SemEval development set with
the `test_set.csv`, creating `data/semeval_test_clean.csv`.


### Running the Hyperparameter search
The hyperparameter search uses only the Chinese data, as we have access to
a train, dev, and test set for this langauge only. As SemEval test data is
not made public, we use the SemEval validation set as a test set.

We follow the majority of hyperparameters from Wu et al. (2023). While they
vary the batch size, we keep the batch size at 8, offering consistency across
models; for larger models the batch size would have to be reduced to run on thec
GPUs (might be possible to run with 16 batch size but not too important).

Following the learning rates used in Wu et al. (2023), for each model we use,
we test the following learning rates:
* 1e-4
* 5e-5
* 3e-5
* 2e-5
* 5e-6.

We only use the seed 555 for the hyperparameter selection process. We also use
only 30 epochs in the hyperparameter search rather than 100 (still with 0.1 warmup epoch ratio).

We select the learning rate hyperparameter for the model by running
the selection process, selecting the learning rate where the validation set
achieves the highest F1-micro score.

The config files for running the hyperparameter searches can be found in
`cfg/hyperparam_search/`.

To run the hyperparameter search:
```
python src/sem_main.py --config="cfg/hyperparam_search/{model_name}/{config_file}"
```


### Running Experiment
Once the learning rate hyperparameter has been selected for the model, 
the rest of the experiments can be run, training on only the Chinese framing
data, only the SemEval training data or the SemEval + Chinese framing training
data.

For each training set we use, we train with the same training hyperparameters but
across seeds 555, 666, and 777, allowing us to assess the variation in test results.

All error evaluation will be performed with the seed 555.

The config files for running the main experiments can be found in
`cfg/`. To run the experiments, run each individual configuration:
```
python src/sem_main.py --config="cfg/{model_name}/{setting}/{config_file}"
```

Before running these experiments, the text should be preprocessed:
* Newline character is added between the news title and body;
* Duplicate sentences occurring consecutively are removed;
* Hyperlinks to websites and images are removed;
* Strings detailing author biographies (e.g. names and affiliations) are removed.


### Analysing Results

The `src/analysis` folder contains 5 files:
* `calculate_results.py`, used for calculating results from the results file across multiple seeds;
* `error_analysis.py`, used to carry out error analysis, with options of creating a co-occurrence
matrix or getting a classification report;
* `find_tricky_samples.py`, used to find particularly difficult samples that one model performs well at
classifying but another does not;
* `results_from_csv.py`, used similarly to `calculate_results.py`, this time calculating the results from
the set of predictions rather than from a results txt file this only calculates the results for the single
predictions file, rather than across all three seeds (555, 666, 777);
* `utils.py` containing utility functions used in the other scripts.


#### Calculating Results
To calculate results using `calculate_results.py`, you need to provide the base directory of your results
(this will be the output_dir_name in your config when you ran the experiments). You must then provide
the setting (either `all`, `chinese_only`, or `sem_only`. Please note that you should use the full path
to the home directory rather than `~`.

An example usage of this script is:
```
python src/analysis/calculate_results.py --base_dir="/home/user/results/xlm-roberta" --setting="all"
```

This will provide you with a summary of results across all languages.



To calculate results using the test_predictions.csv files, you may use `results_from_csv.py`.
An example usage of this is:
```
python src/analysis/results_from_csv.py --data_path="/home/user/results/xlm-roberta/seed_555/all/test_predictions.csv"
```

#### Error analysis
To carry out error analysis, you can use `src/analysis/error_analysis.py`.

To create and save/display a co-occurrence matrix from an individual predictions file:
```
python src/analysis/error_analysis.py --data_path="/home/user/results/xlm-roberta/seed_555/all/test_predictions.csv" --co_occurrence_matrix --lower="true" --save_path="./co_occurrence_matrix.png"
```

In this example, we have loaded the test predictions from `/home/user/results/xlm-roberta/seed_555/all/test_predictions.csv`, set the
lower triangle in the co-occurrence matrix to the number of true co-occurrences (the matrix is symmetric so this allows
us to view extra information), and we have saved this figure to `./co_occurrence_matrix.png`. This will also display the co-occurrence
matrix to you after you run the script. If you omit `--save_path`, your figure will not save. If you wish to save your figure but
not view the figure on generation, use the `--noshow argument`. Additionally, if you would like to restrict
the classification report or the co-occurrence matrix to one language, use the `--lang` argument.


To get an sklearn classification report, containing the precision, recall, and F1 measure for each class, use the `--classification_summary`
argument. An example usage of this would be:
```
python src/analysis/error_analysis.py --data_path="/home/user/results/xlm-roberta/seed_555/all/test_predictions.csv" --classification_summary
```

## Overall Structure

`src/sem_main.py` is the main entry point. This contains the high-level code for training, evaluating and testing the model
using the train, dev, and test splits.

`src/preprocessing/preprocess.py` contains code to select the correct label column given the configuration options of
`train_label_method` and `test_label_method`. Once the correct label is selected, the dataframe containing the text data
(provided by the configuration argument `text_col`) is converted to a dataset, with the text tokenised, ready for use
in model training.

`src/preprocessing/label_generator.py` is simply used to generate binary vector labels given the list of frames seen
in the article, according to the gold standard; 1s represent the frame being present in the article.

`src/trainers/soft_label_topic_trainer.py` contains the model code responsible for training the language models. While
the `SoftLabelTrainer` allows for soft-label training, only hard-label training is used in this study. The use of soft
labels in this task may be an interesting topic for future work as our dataset contains the opinion of multiple
annotators.

All scripts in the `src/dataset_creation/` and `src/analysis/` directories have already been explained through example
in their respective sections.
