

### Step 1: Data Preprocessing:

python code


```
import pandas as pd

# Load the datasets into Pandas DataFrames
sst_df = pd.read_csv("Sst.csv")
chl_df = pd.read_csv("Chl.csv")
fish_df = pd.read_csv("FishName.csv")

# Merge the sst and chl datasets based on common columns
merged_df = pd.merge(sst_df, chl_df, on=["Lat", "Lon", "Month", "Season", "Time"])

# Merge the resulting dataset with the fish dataset based on temperature range groups
merged_df = pd.merge(merged_df, fish_df, on="Fish Temp In Category")

# Clean the data by removing missing or invalid values
merged_df = merged_df.dropna()

```

Note that handling outliers may require additional data cleaning steps beyond what's shown here.

### Step 2: Feature Selection:

python code

```
# Identify the relevant features for the prediction task
selected_features = ["Lat", "Lon", "Month", "Season", "Time", "Sst", "chl", "Fish Temp In Category"]

# Create a new DataFrame with only the selected features
features_df = merged_df[selected_features]

```
Note that the feature selection process may require domain knowledge and exploration of the data to determine which features are most relevant for the prediction task.


### Step 3: Split Data into Train and Test Sets:

python code

```
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets with a ratio of 70:30
X = features_df.drop("Fish Temp In Category", axis=1)
y = features_df["Fish Temp In Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```

Note that we split the data into `X` (features) and `y` (target) variables, then used the `train_test_split()` function from the `sklearn` library to split the data into training and testing sets with a ratio of 70:30. We set the `random_state` parameter to 42 to ensure that the split is reproducible.

### step 4 : code for training a KNN model on the preprocessed data:

python code

```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Select the relevant features
X = preprocessed_data[['Sst', 'Chl', 'Fish Temp In Category', 'Lat', 'Lon']]

# Select the target variable
y = preprocessed_data['Fish Name']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the KNN model
knn = KNeighborsClassifier()

# Define the hyperparameters to search over
param_grid = {'n_neighbors': [3, 5, 7, 9]}

# Use grid search to find the best hyperparameters
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Train the model with the best hyperparameters
knn = grid_search.best_estimator_
knn.fit(X_train, y_train)

```

This code selects the relevant features from the preprocessed data, splits the data into training and testing sets, defines a KNN model, uses grid search to find the best hyperparameters, and finally trains the model with the best hyperparameters.


here's an example code that applies the top 5 algorithms - CNN, KNN, ANN, ARIMA, and SARIMA - on the preprocessed data:

python code

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Split data into features and target
X = df.drop("fish_species", axis=1)
y = df["fish_species"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply CNN
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], 1, 1)))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1)), y_train)

# Apply KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Apply ANN
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001,
                          solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
ann_model.fit(X_train, y_train)

# Apply ARIMA
arima_model = ARIMA(y_train, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Apply SARIMA
sarima_model = SARIMAX(y_train, order=(1, 0, 0), seasonal_order=(1, 1, 1, 12))
sarima_model_fit = sarima_model.fit()

# Evaluate models on testing set
cnn_score = cnn_model.evaluate(X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1)), y_test)[1]
knn_score = knn_model.score(X_test, y_test)
ann_score = ann_model.score(X_test, y_test)
arima_score = arima_model_fit.score(X_test, y_test)
sarima_score = sarima_model_fit.score(X_test, y_test)

print("CNN Accuracy: ", cnn_score)
print("KNN Accuracy: ", knn_score)
print("ANN Accuracy: ", ann_score)
print("ARIMA Accuracy: ", arima_score)
print("SARIMA Accuracy: ", sarima_score)

```

This code loads the preprocessed data into a Pandas DataFrame, splits it into features and target, and then splits it into training and testing sets using a 70:30 ratio. The data is then standardized using the StandardScaler from scikit-learn.

The code applies the top 5 algorithms - CNN, KNN, ANN, ARIMA

### Step 5: Evaluate the Model Use the testing set to evaluate the model's performance and calculate metrics such as accuracy, precision, recall, and F1-score to measure how well the model is performing.

Here's an  code for evaluating the model using the testing set:

python code

```
# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-Score: {:.2f}%".format(f1 * 100))

```

In this example, we are using the `accuracy_score`, `precision_score`, `recall_score`, and `f1_score` functions from the `sklearn.metrics` module to calculate the evaluation metrics. These functions take the actual labels (`y_test`) and the predicted labels (`y_pred`) as inputs and return the corresponding metric values.

Note that we are using the `weighted` average method for calculating precision, recall, and F1-score, which takes into account the imbalance in the class distribution. You can also use other average methods such as `micro` and `macro` depending on your specific requirements.

### Step 6: Improve the Model If the model's performance is not satisfactory, you can try to 
			improve it by tuning the hyperparameters, adding more features, or using a different algorithm. You can also try to combine two or more algorithms to improve the model's performance.

Here's some code for improving the model by tuning hyperparameters using GridSearchCV:

python code

```
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
params = {'n_estimators': [50, 100, 200], 
          'max_depth': [5, 10, None], 
          'min_samples_split': [2, 5, 10]}

# Create a GridSearchCV object
grid_search = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy')

# Fit the model using the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Use the best parameters to create a new model
model = RandomForestClassifier(n_estimators=best_params['n_estimators'], 
                                max_depth=best_params['max_depth'], 
                                min_samples_split=best_params['min_samples_split'])

# Train the new model using the training data
model.fit(X_train, y_train)

# Evaluate the new model using the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

```

This code tunes the hyperparameters of a Random Forest Classifier using GridSearchCV and creates a new model with the best parameters. Then it trains the new model using the training data and evaluates its performance using the testing data. Finally, it prints the accuracy, precision, recall, and F1-score of the new model. You can modify this code to tune the hyperparameters of other algorithms or to add more features to the model.

### Step 7: Make Predictions Once the model is trained and evaluated, you can use it to make predictions on new data. 
Given the sea surface temperature, chlorophyll concentration, and latitude/longitude coordinates, the model can predict the location where a particular fish species can be found.

Here's an code for making predictions using a trained machine learning model:

python code

```
# Load the new data for prediction
new_data = pd.read_csv("new_data.csv")

# Preprocess the new data
# ...

# Use the trained model to make predictions
predictions = model.predict(new_data)

# Print the predicted locations
print(predictions)

```

Note that you will need to preprocess the new data in the same way as the training and testing data before making predictions. The `predict()` function of the trained model can then be used to make predictions on the new data. The output will be the predicted locations of the fish species based on the features provided in the new data.

### Step 8: Visualize the Results You can create charts and graphs to visualize the model's 
results and gain insights into the relationship between the various features and the location of the fish species.

code for visualizing the results using matplotlib:

python code

```
import matplotlib.pyplot as plt

# Plot predicted vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')

# Plot feature importance
feature_importance = model.feature_importances_
feature_names = X.columns.values
sorted_idx = feature_importance.argsort()

plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()

```

This code plots the predicted vs actual values and feature importance for the trained model. You can customize the plots and add more visualizations as necessary.