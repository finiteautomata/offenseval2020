# OffensEval 2020


A FAQ is available to help you with the most frequent questions: https://sites.google.com/site/offensevalsharedtask/faq

More information about evaluation will be sent via e-mail in due time.

## Instructions

`python >= 3.6` and `pipenv` are needed

0. Install requirements

```
pipenv shell
pipenv sync
```

1. Get data

```
./bin/get_data.sh
./bin/get_translations.sh
```

Now, in `data/<LANGUAGE>` you will have the datasets for each language.

2. Install `jupyter` extension for `ipywidgets`

```
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

3. Generate samples and dev datasets

```
python bin/generate_samples.py --sample_frac 0.01
python bin/split_datasets.py --frac 0.2
```

4. Train BERT

Naming convention:

`bert_model.lang.pt`

`bert_model` can be `[bert_uncased, bert_cased]`

```
python bin/train_bert.py bert_cased models/bert_cased.en.pt \
--train_path data/English/task_a_distant.sample.tsv \
--dev_path data/olid/olid-training-v1.0.tsv \
--test_path data/olid/test_a.tsv
```

To test everything is working ok (using a micro dataset) run

```
python bin/train_bert.py bert_cased models/bert_test.en.pt \
--train_path data/English/task_a_distant.xsmall.tsv \
--dev_path data/English/task_a_distant.xsmall.tsv \
--test_path data/English/task_a_distant.xsmall.tsv
```


### Tests

We use `pytest` for our tests. Just run

```
pytest tests/
```

### Data

Thanks for registering for OffensEval 2020!

All training sets are available. Please visit the following URLs.

- Arabic, Danish, Greek, and Turkish: https://www.dropbox.com/sh/qfootyofgiywjjl/AABJefEt7p3wEOH5ohlG3t9ta?dl=0

- English: https://drive.google.com/drive/folders/1-Jn13sc3q-WUp3TZWegv9Yf6UNlICtpu?usp=sharing

Password: sem2020-t12

A FAQ is available to help you with the most frequent questions: https://sites.google.com/site/offensevalsharedtask/faq

More information about evaluation will be sent via e-mail in due time.
