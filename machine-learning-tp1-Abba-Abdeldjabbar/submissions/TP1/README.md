# Student 01

**Name:** Abdeldjabbar

My solution for Machine Learning TP1.

File: `TP1_Solutions.ipynb`

## Exercise 1: Iris Dataset

### Step 1 - Load the data

I used pandas to open `Iris_Dataset.csv` and saved the data in `D`.

### Step 2 - Explore the data

I used `info()`, `describe()`, `shape`, `columns` and `head()` to look at the dataset.

The dataset has 150 rows and 5 columns. Four columns contain numbers and the last one, `species`, contains the flower types.

### Step 3 - Select columns

I selected `sepal_length` and `petal_width` and saved them in `selected_features`.

I displayed the first 20 rows as requested.

### Step 4 - Copy the data

I made a copy called `D_copy` using `D.copy()`.

Both datasets have the same shape `(150, 5)`. The copy is useful because I can try changes without changing the original data.

### Step 5 - Split the data

I used `sepal_length` and `petal_width` for `X`, and `species` for `y`.

I split the data into training and test sets using `train_test_split()`. I used `random_state=42` and `stratify=y`.

### Step 6 - Check the split

The results were:

- Total rows: 150
- Training set: 120 rows (80%)
- Test set: 30 rows (20%)

Each flower type represents one third of both sets.

The test set should stay separate because it is used to check how well a model works on data it did not train on. Using it during training can lead to data leakage.



## Exercise 2: Exploring Text Data

### Step 1 - Read the text file

I opened `txt_file.txt` using UTF-8 encoding and read its contents.
The file contains six lines of comma-separated values with labels
such as `c1` and `c2`.

### Step 2 - Read the file line by line

I used a `for` loop to print the file one line at a time.
I used `rstrip("\n")` to remove the newline character at the end
of each line.

### Step 3 - Load 20 Newsgroups

I loaded the training data using `fetch_20newsgroups()` from
`sklearn.datasets`. I used the default categories.


### Step 4 - Count documents and labels

I checked the number of documents, target labels, and classes.

The training dataset contains 11314 documents and 20 classes.
Each document has one target label.


### Step 5 - Display the first document

I displayed the first document using `newsgroups.data[0]` and checked
its numeric label using `newsgroups.target[0]`.

The document contains text, and the target is a number representing
its class.


### Step 6 - Display class names

I displayed the available class names using `target_names`.
I also used the first document's numeric label to find its class name.

The target is a number, and `target_names` helps us understand
which class that number represents.


## Exercise 3: Missing Values

### Step 1 - Load the employee data

I loaded `employees.csv` into a DataFrame called `employees`.
I checked the column names, data types, and first five rows.

The dataset contains employee information such as names, salaries,
gender, and teams.

### Step 2 - Count Missing Values

I used `employees.isna().sum()` to count the missing values
in each column.

This helped me find which columns have empty values and how many
are missing.


### Step 3 - Visualize Missing Values

I used Seaborn to create a heatmap of the missing values.
The plot helps me see which columns contain empty values.
I added a title and labels to both axes.

### Step 4 - Add a row with missing values

I made a copy of the employee data and added one new row.
I used `-` for the first name and `na` for the gender.
The other six fields contain missing values.

I kept the original dataset unchanged.


### Step 5 - Recount Missing Values

I counted the missing values again and compared them with the original data.

The six fields where I added `np.nan` have one more missing value.
Pandas does not count `-` and `na` as missing because they are strings.


### Step 6 - Replace missing-value markers

I replaced `-` and `na` with `np.nan` in the copied dataset.
After the replacement, pandas can detect them as missing values.

I kept the original employee data unchanged.


### Step 7 - Handle Missing Values

I used `SimpleImputer` to fill missing values in selected columns.
I used the median for `Salary` and `Bonus %`, and the most frequent
value for `Gender`, `Senior Management`, and `Team`.

I left names and date/time fields unchanged because filling them
with arbitrary values could be misleading.


### Step 8 - Compare the results

I compared the missing values before and after imputation.

The missing values in the selected numerical and categorical columnswere filled.
Some missing values remain in the columns I did not process.

Imputation uses estimated values, so the filled values may not bethe real values.
For a prediction model, the imputer should be fittedon training data only.


## Exercise 4: Scaling and Encoding

### Part A - Step 1: Scaling

I created the numerical matrix from the exercise and tried
StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler,and normalize.

The methods gave different results because each one scales
or normalizes the data in a different way.


### Part A - Step 2: Binarization

I used `Binarizer` with a threshold of 0.5 to convert the
original matrix into 0s and 1s.

Values greater than 0.5 became 1, while the other values became 0.
Changing the threshold can change the result.


### Part A - Discussion

The five methods scale or normalize the data in different ways.
The first four work on features, while `normalize` works on samples
by default.

Scaling can help KNN, SVM, and gradient-based algorithms.
Decision trees usually do not require it.



### Part B - Step 1: Ordinal Encoding

I used `OrdinalEncoder` to convert the three input features into numbers.
I also displayed the categories learned for each column using `categories_`.

The numbers are category codes. They do not necessarily represent
a real order between the categories.


### Part B - Step 2: Encode the Target

I used `LabelEncoder` to convert the target class names into numbers.
I displayed the class names and their numeric labels in a table.

The numbers represent the classes and do not show a real ranking.


### Part B - Step 3: One-Hot Encoding

I used `OneHotEncoder` to represent each target class with a separate
column containing 0s and 1s.

Unlike LabelEncoder, this method does not represent the classes
with a single numerical code.


## Conclusion

In this TP, I practiced exploring data, handling missing values,
scaling numerical data, and encoding categorical variables.

I also learned why we split data into training and test sets
and why we should avoid data leakage.
