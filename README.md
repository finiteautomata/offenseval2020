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

### Data

Thanks for registering for OffensEval 2020!

All training sets are available. Please visit the following URLs.

- Arabic, Danish, Greek, and Turkish: https://www.dropbox.com/sh/qfootyofgiywjjl/AABJefEt7p3wEOH5ohlG3t9ta?dl=0

- English: https://drive.google.com/drive/folders/1-Jn13sc3q-WUp3TZWegv9Yf6UNlICtpu?usp=sharing

Password: sem2020-t12

A FAQ is available to help you with the most frequent questions: https://sites.google.com/site/offensevalsharedtask/faq

More information about evaluation will be sent via e-mail in due time.
