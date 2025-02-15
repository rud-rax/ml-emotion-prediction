# ml-emotion-prediction

1. Load the dataset as Pandas dataframe.
2. Clean the dataset ( For example - check and handle NULL values and N/A values). (at least one technique)
3. Visualize the dataset and come up with some insights (at least one visualization)
4. Preprocess the dataset using standard preprocessing techniques.
5. Save the preprocessed data as a CSV file.
---
6. Load the preprocessed data and split it into Train, Validation, and Test sets. (60:20:20)
7. Train at least three regression models with different complexities with the train set. 
8. Use the Validation Set to do hyperparameter tuning if required.
9. After finding the hyperparameters, merge the train and validation sets to be your new train set and train your model again on train+validation sets. 
10. Do K-Fold Cross-validation with k = 5. Test the models using the test set and report metrics like RMSE and MSE.
11. Do the above steps with feature selection and report metrics (at least one FS technique).
---
12. Compare the metrics before and after feature selection.

## Feature Selection
To begin with, I carefully examined the dataset and identified columns that were not essential for training the model. These columns were removed to ensure that only relevant features were used, thereby reducing noise and improving the model’s performance. Once the data was refined, I employed multiple feature selection techniques to identify the most significant predictors.

First, I utilized a **Correlation Matrix** to analyze the relationships between different features and the target variable. This helped in identifying highly correlated features that could potentially introduce multicollinearity, which might negatively impact the model’s interpretability and performance. After that, I applied Recursive Feature Elimination (RFE), a method that iteratively removes less important features and ranks them based on their importance to the model’s predictions. This ensured that only the most relevant features were retained.

Additionally, I leveraged the **XGBoost Regressor** (XGB Regressor), a powerful gradient boosting algorithm, to rank features based on their predictive power. This technique provided valuable insights into which features contributed most significantly to the target variable. Furthermore, to ensure data quality, I used the Isolation Forest algorithm to detect and remove outliers. Outliers can distort the learning process and reduce model accuracy, so their removal helped improve overall performance and robustness of the model.


## Model Validation
For model training, I experimented with multiple regression models to determine the most effective approach for predicting the target variable. Initially, I used Linear Regression as a baseline model to establish a point of comparison. However, given the non-linearity in the data, I explored machine learning models such as **K-Nearest Neighbors (KNN), Random Forest Regressor, and Gradient Boosting Regressor**.

K-Nearest Neighbors (KNN) was tested with different values of neighbors and weight functions to optimize performance. I fine-tuned hyperparameters like the number of neighbors and distance metrics to enhance accuracy. The Random Forest Regressor was validated with various numbers of estimators and different criterion functions (such as absolute error and Friedman's MSE) to determine the most robust configuration. Additionally, Gradient Boosting Regressor was employed with different loss functions, including absolute error, Huber, and quantile loss, to handle the non-linearity in the dataset effectively.

To ensure a reliable evaluation, I used cross-validation techniques to validate each model's performance. This included k-fold cross-validation, which helped in assessing model stability and generalization. Additionally, I measured performance using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared scores. This systematic approach enabled me to select the most suitable model for the given dataset.




| Model                         | Variants / Hyperparameters Tested                     | Test RMSE | Test MSE | Test R²  |
|-------------------------------|-----------------------------------------------------|----------|---------|---------|
| **Linear Regression**         | -                                                  | 0.2366   | 0.0560  | 0.8546  |
| **K-Nearest Neighbors**       | n_neighbors (various values), weights (distance-based) | 0.3729 - 0.3673 | 0.1390 - 0.1349 | 0.6390 - 0.6496 |
| **Random Forest Regressor**   | n_estimators (various values), criterion (absolute error, Friedman's MSE) | 0.2139 - 0.2090 | 0.0458 - 0.0437 | 0.8812 - 0.8865 |
| **Gradient Boosting Regressor** | loss function (absolute error, Huber, quantile) | 0.2117 - 0.2094 | 0.0448 - 0.0438 | 0.8836 - 0.8861 |


![21 GradientBoostingRegressor(loss='huber')](https://github.com/user-attachments/assets/418c40e9-5044-4290-9683-27f1900c4d56)
![18 RandomForestRegressor(criterion='friedman_mse', random_state=42)](https://github.com/user-attachments/assets/043ab750-0d32-4614-9226-08c3af21edcb)
![4 KNeighborsRegressor(weights='distance')](https://github.com/user-attachments/assets/56a31c94-4dd8-4eeb-83ae-356ff5903603)
![0 LinearRegression()](https://github.com/user-attachments/assets/ab40e303-3407-4a1c-a1aa-c935d2d84fec)

