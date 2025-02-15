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
