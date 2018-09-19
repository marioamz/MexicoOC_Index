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

## Navigate to iPython3

The algorithms run from iPython3 on the terminal. From the directory where you've saved the four python files and the data subfolder that includes the excel files with data you'll be working with segregated by year, please run the following command on your terminal.

```
ipython3
```

You should see an iPython environment open in your terminal.

## Impute missing variables

The first step in the program is to impute missing variables in the data. In order to do so, we need to import the missing data code into ipython with the following command.

```
import missing_data_code as mdc
```

Once the code is imported, we run the go function which has the following parameters:
- excelname = the name of the excel file of the data we're working with (ie: 'data/2010.xlsx')
- method = the method we want to use to impute missing data, there are several options:
  - 'zero' = imputes all missing values with zeroes
  - 'delete_row' = deletes all rows with missing values
  - 'delete_column' = deletes all columns with missing values
  - 'mode' = imputes missing values with the mode of the column
  - 'median' = imputes missing values with the median of the column
  - 'mean' = imputes missing values with mean of the column
  - 'linear' = runs a linear regression to estimate the missing values in a given row
  - 'k' = estimates missing values using a k-nearest neighbors approach
- index = the index of the data, which is likely to be the name of the column where all the country names are stores
- knn = the number of nearest neighbors to use, only use if the method you've chosen is 'k'. Otherwise, use None.
- weight = the weight metric for nearest neighbors, only use if the method you've chosen is 'k'. There are three options.
  - 'uniform'
  - 'distance'
  - None: use if you're not employing the 'k' method
- distance = the distance metric for nearest neighbors, only use if the method you've chosen is 'k'. There are four options.
  - 'euclidian'
  - 'manhattan'
  - 'minkowski'
  - None: use if you're not employing the 'k' method





