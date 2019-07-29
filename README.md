## DAS3H

This repository contains the Python code used for the experiments from our EDM 2019 paper: [_DAS3H: Modeling Student Learning and Forgetting for
Optimally Scheduling Distributed Practice of Skills_](https://arxiv.org/abs/1905.06873). Authors: [Benoît Choffin](https://github.com/BenoitChoffin), [Fabrice Popineau](https://github.com/fpopineau), Yolaine Bourda, and [Jill-Jênn Vie](https://github.com/jilljenn).

Code for this repository is partly borrowed from [jilljenn](http://jill-jenn.net/)'s [ktm repository](https://github.com/jilljenn/ktm).

It is recommended to use a virtual environment for running our experiments. In order to use embedding dimensions *d* > 0, [libfm](https://github.com/srendle/libfm) needs to be installed as well:

```
git clone https://github.com/srendle/libfm
cd libfm && git reset --hard 91f8504a15120ef6815d6e10cc7dee42eebaab0f && make all
```

#### Preparing data

Three open access datasets were used for our experiments:
* [ASSISTments 2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) (assistments12)
* Bridge to Algebra 2006-2007 (bridge_algebra06)
* Algebra I 2005-2006 (algebra05)

The two last datasets come from the [KDD Cup 2010 EDM Challenge](http://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp). Datasets need to be downloaded and put inside each corresponding data folder in data. The main dataset (`train` for KDD Cup) should each time be renamed "data" + corresponding extension name.

To preprocess each of the datasets:

```
python prepare_data.py --dataset <dataset codename> --min_interactions 10 --remove_nan_skills
```

#### Encoding sparse features

To encode sparse features on which the ML models will train, `encode.py` is used. The preprocessed dataset is automatically selected. For instance, DAS3H is "users, items, skills, wins, attempts, tw_kc":

```
python encode.py --dataset <dataset codename> --users --items --skills --wins --attempts --tw_kc
```

|   | users | items | skills | wins | fails | attempts | tw_kc | tw_items |
|:-:|:-----:|:-----:|:------:|:----:|:-----:|:--------:|:-----:|:--------:|
| DAS3H | x | x | x | x | | x | x | |
| DASH | x | x | | x | | x | | x |
| IRT/MIRT | x | x | | | | | | |
| PFA | | | x | x | x | | | |
| AFM | | | x | | | x | | |

A faster script for encoding DAS3H time windows is available [here](https://github.com/jilljenn/ktm/blob/master/encode_tw.py#L115).

#### Running the models

Code for running the experiments is in `das3h.py`. For instance, for performing cross-validation for DAS3H with embedding dimension *d*=5, on ASSISTments12:

```
python das3h.py data/assistments12/X-uiswat1.npz --dataset assistments12 --d 5 --users --items --skills --wins --attempts --tw_kc
```

## Appendix: complete metrics tables
Algebra 2005-2006 (PSLC DataShop) dataset:

| model | dim | AUC | ACC | NLL |
|:-:|:-----:|:-----:|:------:|:----:|
| DAS3H | 0 | **0.826** ± 0.003 | **0.815** ± 0.007 | **0.414** ± 0.011 |
| DAS3H | 5 | **0.818** ± 0.004 | **0.812** ± 0.007 | **0.421** ± 0.011 |
| DAS3H | 20 | **0.817** ± 0.005 | **0.811** ± 0.004  | **0.422** ± 0.007 |
| DASH | 5 | 0.775 ± 0.005 | 0.802 ± 0.010 | 0.458 ± 0.012 |
| DASH | 20 | 0.774 ± 0.005 | 0.803 ± 0.014 | 0.456 ± 0.017 |
| DASH | 0 | 0.773 ± 0.002 | 0.801 ± 0.004 | 0.454 ± 0.006 |
| IRT | 0 | 0.771 ± 0.007 | 0.800 ± 0.009 | 0.456 ± 0.015 |
| MIRTb | 20 | 0.770 ± 0.007 | 0.800 ± 0.006 | 0.460 ± 0.007 |
| MIRTb | 5 | 0.770 ± 0.004 | 0.800 ± 0.008 | 0.459 ± 0.011|
| PFA | 0 | 0.744 ± 0.004 | 0.782 ± 0.003 | 0.481 ± 0.004 |
| AFM | 0 | 0.707 ± 0.005 | 0.774 ± 0.004 | 0.499 ± 0.006 |
| PFA | 20 | 0.670 ± 0.010 | 0.748 ± 0.005 | 1.008 ± 0.047 |
| PFA | 5 | 0.664 ± 0.010 | 0.735 ± 0.013 | 1.107 ± 0.079 |
| AFM | 20 | 0.644 ± 0.005 | 0.750 ± 0.005 | 0.817 ± 0.076 |
| AFM | 5 | 0.640 ± 0.007 | 0.742 ± 0.009 | 0.941 ± 0.056 |

ASSISTments 2012-2013 dataset:

| model | dim | AUC | ACC | NLL |
|:-:|:-----:|:-----:|:------:|:----:|
| DAS3H | 5 | **0.744** ± 0.002 | **0.737** ± 0.001 | **0.531** ± 0.001 |
| DAS3H | 20 | **0.740** ± 0.001 | **0.736** ± 0.002 | **0.533** ± 0.003 |
| DAS3H | 0 | **0.739** ± 0.001 | **0.736** ± 0.001 | **0.534** ± 0.002 |
| DASH | 0 | 0.703 ± 0.002 | 0.719 ± 0.003 | 0.557 ± 0.004 |
| DASH | 5 | 0.703 ± 0.001 | 0.720 ± 0.001 | 0.557 ± 0.001 |
| DASH | 20 | 0.703 ± 0.002 | 0.720 ± 0.002 | 0.557 ± 0.002 |
| IRT | 0 | 0.702 ± 0.001 | 0.719 ± 0.001 | 0.558 ± 0.001 |
| MIRTb | 20 | 0.701 ± 0.001 | 0.720 ± 0.001 | 0.558 ± 0.001 |
| MIRTb | 5 | 0.701 ± 0.002 | 0.719 ± 0.001 | 0.558 ± 0.001 |
| PFA | 5 | 0.669 ± 0.002 | 0.709 ± 0.002 | 0.577 ± 0.002 |
| PFA | 20 | 0.668 ± 0.002 | 0.709 ± 0.003 | 0.578 ± 0.003|
| PFA | 0 | 0.668 ± 0.002 | 0.708 ± 0.001 | 0.579 ± 0.002 |
| AFM | 5 | 0.610 ± 0.001 | 0.699 ± 0.002 | 0.597 ± 0.001 |
| AFM | 20 | 0.609 ± 0.001 | 0.699 ± 0.003 | 0.597 ± 0.003 |
| AFM | 0 | 0.608 ± 0.002 | 0.697 ± 0.002 | 0.598 ± 0.002 |

Bridge to Algebra 2006-2007 (PSLC DataShop):

| model | dim | AUC | ACC | NLL |
|:-:|:-----:|:-----:|:------:|:----:|
| DAS3H | 5 | **0.791** ± 0.005 | **0.848** ± 0.002 | **0.369** ± 0.005 |
| DAS3H | 0 | **0.790** ± 0.004 | **0.846** ± 0.002 | **0.371** ± 0.004 |
| DAS3H | 20 | 0.776 ± 0.023 | 0.838 ± 0.019 | 0.387 ± 0.027 |
| DASH | 0 | 0.749 ± 0.002 | 0.840 ± 0.005 | 0.393 ± 0.007 |
| DASH | 20 | 0.747 ± 0.003 | 0.840 ± 0.001 | 0.399 ± 0.002 |
| IRT | 0 | 0.747 ± 0.002 | 0.839 ± 0.004 | 0.393 ± 0.007 |
| DASH | 5 | 0.747 ± 0.003 | 0.840 ± 0.002 | 0.399 ± 0.002 |
| MIRTb | 5 | 0.746 ± 0.002 | 0.840 ± 0.004 | 0.398 ± 0.006 |
| MIRTb | 20 | 0.746 ± 0.004 | 0.839 ± 0.005 | 0.399 ± 0.007 |
| PFA | 20 | 0.746 ± 0.003 | 0.839 ± 0.002 | 0.397 ± 0.004 |
| PFA | 5 | 0.744 ± 0.007 | 0.838 ± 0.003 | 0.402 ± 0.007 |
| PFA | 0 | 0.739 ± 0.003 | 0.835 ± 0.005 | 0.406 ± 0.008 |
| AFM | 5 | 0.706 ± 0.002 | 0.836 ± 0.003 | 0.411 ± 0.004 |
| AFM | 20 | 0.706 ± 0.002 | 0.836 ± 0.003 | 0.412 ± 0.004 |
| AFM | 0 | 0.692 ± 0.002 | 0.833 ± 0.004 | 0.423 ± 0.006 |
