
## PRGC 2018
The goal of Pattern Recognition Grand Challenge (PRGC) 2018 is to predict a subject's emotional state ($Y$) from fMRI image ($X$) of his or her brain.
In the PINES dataset below, 183 subjects' fMRI images ($X$) and emotional states($Y$) are provided.
Build a model to predict a subject's emotional state ($Y$) from an fMRI image ($X$).

## Refences
* See [here](http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002179) for a breif introduction about this study.
* For more detail, read the Method section of the [original article](http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002180)


## Picture Induced Negative Emotion Signature (PINES) Dataset

### 1. fMRI data
https://neurovault.org/collections/503/

This dataset contains single trial responses to 30 images from the International Affective Picture Set from 182 subjects. These data were used to train the Picture Induced Negative Emotion Signature described in [Chang et al., 2015](http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002180). 

The details of the experiment can be seen in Gianaros et al., 2014 (doi:10.1016/j.biopsych.2013.10.012).
* Negative photographs (Pictures: 2053, 3051, 3102, 3120, 3350, 3500, 3550, 6831, 9040, 9050, 9252, 9300, 9400, 9810, 9921) depicted bodily illness and injury (10 photographs), acts of aggression (2 photographs), members of hate groups (1 photograph), transportation accidents (1 photograph) and human waste (1 photograph). 
* Neutral photographs (Pictures: 5720, 5800, 7000, 7006, 7010, 7040, 7060, 7090, 7100,7130, 7150, 7217, 7490, 7500, 9210) depicted inanimate objects (10 photographs) or neutral scenes (5 photographs).

### 2. emotion level data 
* [S1_Data.csv](S1_Data.csv) from https://s3-eu-west-1.amazonaws.com/pstorage-plos-3567654/2129473/S1_Data.csv
* Subject's reponses are stored in column `Rating`.
* Use `Holdout` column to divide the data into training (`Trainig`) and test (`Test`) datasets. 


## Example code
* [loadData.ipynb](loadData.ipynb) - Shows you how to load the dataset

## Installing required packages 

### Let's install  `nibabel` in a new virtual environment

1. Create a virtual environment
    ```
    python3 -m venv ~/ni
    ```

2. Activate the virtual envrinonment
    ```
    source ~/ni/bin/activate
    ```

3. Install Jupyter
    ```
    pip3 install jupyter
    ```
    
4. Install `nibabel` in it
    ```
    pip3 install nibabel
    ```

Once you're done, you can activate the environment (Step 2) and use it.

See http://nipy.org/nibabel/gettingstarted.html for more detals about nibabel.
