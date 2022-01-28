# Activity

This project uses spectrograms and scaleograms to predict human activity (e.g. walking, standing) from sensor information (accelerometer, gyroscope) captured by a smartphone.

The code in this repro relates to the following blog posts:

* [Creating spectrograms and scaleograms for signal classification](https://bea.stollnitz.com/blog/spectrograms-scaleograms/)
(more to come)

## Setup

If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace". Alternatively, you can set up your local machine using the following steps.

Install conda environment:

```
conda env create -f conda.yml
```

Activate conda environment:

```
conda activate activity
```

## Run

Within VS Code, open the `activity/src/1_generate_grams.py` file, and press F5.


## Dataset

The original dataset used can be found [on this page](https://archive-beta.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), and is shared here with the same [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode) license.

Citation:
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.