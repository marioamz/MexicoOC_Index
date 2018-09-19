# Composite Index Builder

This project was built for InSight Crime, a think tank that studies organized crime in Latin America, in order to help them build a composite index that measures the impact of organized crime in Latin American countries. The algorithms included in this repository impute missing variables, explore the data, normalize data, and then weight and aggregate the data into a final composite index value. It does not include other functions essential for building a composite index such as creating binary variables and discretizing values, since the data hasn't been entirely collected yet.

## Installing the necessary tools

In order to run the algorithms, please make sure you have Python 3.6 or 3.7 installed by following the steps at this link: https://www.python.org/downloads/release/python-366/

After installing the latest version of Python, please navigate to your terminal and also install the relevant packages that this program uses with the following commands.

```
pip3 install sklearn
pip3 install missingno
pip3 install scipy
```

## Cloning the git repository

Once the right version of Python is installed and all the packages have been loaded in, it's time to clone the git repository into a new folder on your computer. In that folder, please also include a subfolder called 'data' where you'll keep the data you'll be using to build the index, segregated by year, in excel format.

You can clone the git repository with the following command written on your terminal, from the directory where you'd like to store the text files:

```
git clone https://github.com/marioamz/MexicoOC_Index
```

After running this command, the folder you're working out of should have four new files:

```
missing_data_code.py
explorations_pca_ca.py
normalization.py
weights.py
```

## Imputing Missing Variables

