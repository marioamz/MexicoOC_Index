# Composite Index Builder

This project was built for InSight Crime, a think tank that studies organized crime in Latin America, in order to help them build a composite index that measures the impact of organized crime in Latin American countries. The algorithms included in this repository impute missing variables, explore the data, normalize data, and then weight and aggregate the data into a final composite index value. It does not include other functions essential for building a composite index such as creating binary variables and discretizing values, since the data hasn't been entirely collected yet.

## Step One: Installing the necessary tools

In order to run the algorithms, please make sure you have Python 3.6 or 3.7 installed by following the steps at this link: https://www.python.org/downloads/release/python-366/

After installing the latest version of Python, please navigate to your terminal and also install the relevant packages that this program uses with the following commands.

```
pip3 install sklearn
pip3 install missingno
pip3 install scipy
```

## Step Two: Cloning the git repository

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

## Step Three: Navigate to iPython3

The algorithms run from iPython3 on the terminal. From the directory where you've saved the four python files and the data subfolder that includes the excel files with data you'll be working with segregated by year, please run the following command on your terminal.

```
ipython3
```

You should see an iPython environment open in your terminal.

## Step Four: Impute missing variables

The first step in the program is to impute missing variables in the data. In order to do so, we need to import the missing data code into ipython with the following command.

```
import missing_data_code as mdc
```

Once the code is imported, we'll use the go function which has the following parameters:

```
mdc.go_missing(excelname, method, index, knn, weight, distance)
```

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

### Examples of imputing missing data

If you want to estimate missing variables for your 2012 data with the mean of the columns, and the column name of the country column is pais, you would run the following command in ipython.

```
mdc.go_missing('data/2012.xlsx', 'mean', 'pais', None, None, None)
```

If instead you want to estimate missing variables for your 2014 data with a k-nearest neighbors approach, and the column name of the country column is paises, you would run the following command in ipython.

```
mdc.go_missing('data/2014.xlsx', 'k', 'paises', 5, 'uniform', 'minkowski')
```

Both of these commands would estimate missing variables. The first would do so by taking the mean of the columns with missing variables, and the second by employing a k-nearest neighbors approach that takes into account the five nearest neighbors according to a uniform weighting scheme and a minkowski distance.

### Output of imputing data

The output from this command will be a new excel sheet in your data subfolder called **imputed.xlsx**. This excel sheet has the estimated variables according to the method selected.

## Step Five: Explore the data

The next step in the program is to explore the data in order to find outliers, explore correlations, understand how variables group together under principal component analysis, and do a cluster analysis which results in an initial grouping of countries. 

In order to explore the data, you need to import the exploration_pca_ca.py file into iPython with the following command.

```
import exploration_pca_ca as epc
```

Once the code is imported, we'll run the go function to explore the data fully. The go function has several parameters.

```
epc.go_explore(excelname, components, index1, index2, index3, buckets, year)
```

- excelname: the name of the excel file with the recently imputed data. It should be called 'data/imputed.xlsx'
- components: the number of principal component vectors you want to reduce dimensions to (5, 10, or 15)
- index1: the name of the column in the excel file that contains all the country names (ie 'paises', 'pais', 'countries')
- index2: the name of the column in the excel file that contains the year (ie 'año')
- index3: the name of the column that you want to create with the composite scores, typically 'scores'.
- buckets: the number of buckets you want to divide the composite index into, in this case 5.
- year: the year for which you're doing the analysis (ie 2012)

### Examples of exploring data

If you have an imputed excel file for 2014 where the countries column is named 'paises' and the year column is named 'years', and you want to reduce the dimensionality to 10 components and also break down results into scores of 1, 2, 3, 4, 5, then you would run the following command in iPython.

```
epc.go_explore('data/imputed.xlsx', 10, 'paises', 'years', 'scores', 5, 2014)
```

### Output of exploring data

This algorithm produces a robust output. It creates two new subfolders: graphs and results. 

In the data subfolder, it adds a new excel file.
- loadings.xlsx: this is a file that shows how much variance every variable contributes to a principal component vector. This allows you to see if variables are grouping together in ways that make sense given what you want to measure.

In the graphs folder, it will produce the a series of png files:

- boxplot_columnname_year.png: these files include boxplot images for all columns in the imputed excel file. This is useful in that it lets you see which columns have outliers, which then informs your decisions to discretize, turn to binary, or get rid of those observations entirely.
- 2DPCA_year.png: this file is a graph that captures the distribution for the two components that best explain the variance in the data. It's a first glance at how countries are grouping together, which helps you understand if the data you have is adequate enough to explain the phenomenon you want to explain.
- correlations.png: this file is a correlation heatmap, it shows which columns are positively and negatively correlated with each other. With this information, you can determine whether it's necessary to drop variables that are highly and positively correlated.
- Explained_Variance_year.png: this file is a graph of how much variance is explained by the number of components you chose (10, in the example above). You'd want it to be greater than 90%.
- summary_stats.txt: this text file includes summary statistics for every single column in the excel: mean, median, std, frequency, and more.

In the results folder, it will produce an unsupervised attempt at getting a final, composite index.

- KMeans.xlsx: this is an excel file that groups countries together using an unsupervised kmeans algorithm. If countries are grouping together in ways you would expect, then it means the underlying data is useful.
- affinity_propagation.xlsx: this is an excel file that groups countries together using an unsupervised affinity propagation algorithm. If countries are grouping together in ways you would expect, it means the underlying data is useful.

## Step Six: Normalize the data

The next step in the model is to normalize the data to make it comparable across columns. In order to expore the data, you need to import the normalization.py code into iPython with the following command.

```
import normalization as norm
```

The go function has several parameters.

```
norm.go_normalize(excelname, index, method)
```
-excelname: the excel file with the imputed data, likely 'data/imputed.xlsx'.
-index: the name of the country column in the excel file, and the name of the year column in the excel file. It should be passed in as a list: ['country', 'year']
- method: the method you want to use to normalize the data. There are three options:
  - 'z': normalizes the data by taking the z-score of each observation in each column.
  - 'rank': normalizes the data by ranking each observation in a column.
  - 'categorical': creates categories to normalize the data by assigning values from 1-5 to each observation in a column.
  
### Examples of normalizing data

If I my imputed excel file ('data/imputed.xlsx') had 'paises' as the country column name and 'years' as the year column name, and I wanted to normalize data by ranking it, I would run the following command in iPython.

```
norm.go_normalize('data/imputed.xlsx', ['paises', 'years'], 'rank')
```

### Output of normalizing data

The output from running the go_normalize function is fairly straightforward. In the data subfolder it adds a new excel file.
- normalized.xlsx: this is an excel sheet with all the data normalized.

## Step Seven: Building the composite index!

The last step of the model is to weight and aggregate everything to get the final composite index. In order to run this code, you need to import weights.py into iPython with the following command.

```
import weights.py as wts
```

The go function has several parameters.

```
wts.go_weights(excelname, method, components, index1, index2, year)
```
- excelname: the excel file with the normalized data, typically 'data/normalized.xlsx'
- method: the method by which you want to aggregate the results. There are two options:
  - 'add': adds the weighted scores to get the composite
  - 'geometric': takes the geometric mean of a row to get the composite
- components: the number of components for the principal component analysis, which is how weights are calculated in this model (5, 10, 15)
- index1: the name of the country column in the excel sheet (ie 'paises')
- index2: the name of the year column in the excel sheet (ie 'years')
- year: the year for which you're doing the analysis (ie 2012)

### Example of weighting and aggregating

Let's say you're using the 'data/normalized.xslx' excel file, that you want to weight your variables using 10 components, and that you want to deploy a geometric mean aggregation. Furthermore, let's say you're examining 2015 data and that your country column name is 'paises' and year column name is 'XXX'. You would type in the following command into ipython.

```
wts.go_weights('data/normalized.xlsx', 'geometric', 10, 'paises', 'XXX', 2015)
```

### Output of weighting and aggregating

The output from the go_weights function is the final result. In the results subfolder, you'll find a new excel file called **geometric_aggregation.xlsx** if you chose 'geometric' as the method; or **additive_aggregation.xlsx** if you chose 'add' as the method. This excel file is the composite index score.

