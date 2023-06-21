
-   One-hot Encoding using:
    -   Python’s category_encoding library
    -   Scikit-learn preprocessing
    -   Pandas' get_dummies
-   Binary Encoding
-   Frequency Encoding
-   Label Encoding
-   Ordinal Encoding


## What is Categorical Data?

Categorical data is a type of data that is used to group information with similar characteristics, while numerical data is a type of data that expresses information in the form of numbers.

Example of categorical data: **gender**

**Why do we need encoding?**

-   Most machine learning algorithms cannot handle categorical variables unless we convert them to numerical values
-   Many algorithm’s performances even vary based upon how the categorical variables are encoded

**Categorical variables can be divided into two categories:**

-   Nominal: no particular order
-   Ordinal: there is some order between values

![[categorical_nomi_ordi.png]]

# Method 1: Using Python’s Category Encoder Library

category_encoders is an amazing Python library that provides 15 different encoding schemes.

**Here is the list of the 15 types of encoding the library supports:**

-   One-hot Encoding
-   Label Encoding
-   Ordinal Encoding
-   Helmert Encoding
-   Binary Encoding
-   Frequency Encoding
-   Mean Encoding
-   Weight of Evidence Encoding
-   Probability Ratio Encoding
-   Hashing Encoding
-   Backward Difference Encoding
-   Leave One Out Encoding
-   James-Stein Encoding
-   M-estimator Encoding
-   Thermometer Encoder

# Method 2: Using Pandas' Get Dummies

```
pd.get_dummies(data,columns=["gender","city"])
```

![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_7.png)  

We can assign a prefix if we want to, if we do not want the encoding to use the default.

```
pd.get_dummies(data,prefix=["gen","city"],columns=["gender","city"])
```

  

![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_6.png)  


# Method 3: Using Scikit-learn

Scikit-learn also has 15 different types of built-in encoders, which can be accessed from sklearn.preprocessing.

## Scikit-learn One-hot Encoding

Let's first get the list of categorical variables from our data:

```
s = (data.dtypes == 'object')
cols = list(s[s].index)


from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore',sparse=False)
```

Applying on the gender column:

```
data_gender = pd.DataFrame(ohe.fit_transform(data[["gender"]]))

data_gender
```

  

![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_9.png)  


Applying on the city column:

```
data_city = pd.DataFrame(ohe.fit_transform(data[["city"]]))

data_city
```


![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_8.png)  

   
Applying on the class column:

```
data_class = pd.DataFrame(ohe.fit_transform(data[["class"]]))

data_class
```


![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_11.png)  


This is because the class column has 4 unique values.

Applying to the list of categorical variables:

```
data_cols = pd.DataFrame(ohe.fit_transform(data[cols]))
data_cols
```
  

![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_10.png)  
 

Here the first 2 columns represent gender, the next 4 columns represent class, and the remaining 2 represent city.

## Scikit-learn Label Encoding

In label encoding, each category is assigned a value from 1 through N where N is the number of categories for the feature. There is no relation or order between these assignments.

```
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Label encoder takes no arguments
le_class = le.fit_transform(data[["class"]])

  

# Comparing with one-hot encoding

data_class
```


![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_11.png)  
 

## Ordinal Encoding

Ordinal encoding’s encoded variables retain the ordinal (ordered) nature of the variable. It looks similar to label encoding, the only difference being that label coding doesn't consider whether a variable is ordinal or not; it will then assign a sequence of integers.

**Example: Ordinal encoding will assign values as Very Good(1) < Good(2) < Bad(3) < Worse(4)**

First, we need to assign the original order of the variable through a dictionary.

```
temp = {'temperature' :['very cold', 'cold', 'warm', 'hot', 'very hot']}
df=pd.DataFrame(temp,columns=["temperature"])
temp_dict = {'very cold': 1,'cold': 2,'warm': 3,'hot': 4,"very hot":5}
df
```



![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_22a.png)  

 

Then we can map each row for the variable as per the dictionary.

```
df["temp_ordinal"] = df.temperature.map(temp_dict)
df
```


![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_13.png)  

 

## Frequency Encoding

The category is assigned as per the frequency of values in its total lot.

```
data_freq = pd.DataFrame({'class'  :['A','B','C','D','A',"B","E","E","D","C","C","C","E","A","A"]})


# Grouping by class column:
fe = data_freq.groupby("class").size()


# Dividing by length:
fe_ = fe/len(data_freq)


# Mapping and rounding off:
data_freq["data_fe"] = data_freq["class"].map(fe_).round(2)
data_freq
```

  

![Figure](https://www.kdnuggets.com/wp-content/uploads/garg_cat_variables_14.png)  


## Mean Encoding: 

Mean Encoding or Target Encoding is one very popular encoding approach followed by Kagglers. Mean encoding is similar to label encoding, except here labels are correlated directly with the target. For example, in mean target encoding for each category in the feature label is decided with the mean value of the target variable on a training data.

The advantages of the mean target encoding are that it **does not affect the volume of the data** and helps in faster learning.

![](https://miro.medium.com/v2/resize:fit:700/1*8lK9mSxuPJ4b9SUXA3dN-A.png)

In this article, we saw 5 types of encoding schemes. Similarly, there are 10 other types of encoding which we have not looked at:

-   Helmert Encoding
-   Mean Encoding
-   Weight of Evidence Encoding
-   Probability Ratio Encoding
-   Hashing Encoding
-   Backward Difference Encoding
-   Leave One Out Encoding
-   James-Stein Encoding
-   M-estimator Encoding
-   Thermometer Encoder

# Which Encoding Method is Best?

There is no single method that works best for every problem or dataset. I personally think that the get_dummies method has an advantage in its ability to be implemented very easily.

If you want to read about all 15 types of encoding, [here is a very good article to refer to](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02).

Here is a cheat sheet on when to use what type of encoding:
 
![[categorical Encoding Cheat-seet.png]]