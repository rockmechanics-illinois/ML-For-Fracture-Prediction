# Machine Learning Models for Fracture Prediction
## Overview
Analyzes geophysical log data and employs various machine learning algorithms to predict fractured depth intervals based on the log data. No datasets are published.

## How to Navigate Code

### Import and Preprocess the LAS files
*	Reads the “for code” sheet in the Feature Map excel sheet, which indicates which LAS the data for each feature should be extracted from. Then, extracts the data accordingly, ensures the depths are recorded at 0.5 feet intervals, removes rows that contain NaN values, and reads another excel sheet (Rock Formation Depths) to add column indicating the rock formation at each depth. Then, it writes the raw data to CSV files, and also applies a min-max normalization to the data and writes that output to additional CSV files.
    *	Python notebook to run: Import and Preprocess LAS Data Using Feat Map.ipynb
    	* Path: REU-Project/Python Notebooks/Prepare Data For Models/)
    *	Where it imports from: LAS Data Files/, Featmap.xlsx, Rock Formation Depths.xlsx
    *	Where it writes to: Raw CSV Files/, Norm CSV Files/

 ### Extract Labels
 * Consulting the FMI Logs Excel sheet, it assigns the fracture status of every depth included in the datalogs. It plots the fractured depths and plots the distribution of fractured vs intact intervals.
    * Python notebook to run: Extract Labels.ipynb 
      * Path: REU-Project/Python Notebooks/Prepare Data For Models/
    * Where it imports from: Rock Formation Depths.xlsx, FMI Logs.xlsx, Raw or Norm CSV Files/
    * Where it writes to: labels/

### Run The Models
* Select which model you would like to run (K-Nearest Neighbors, Logistic Regression, Random Forest, XGBoost Classifier, XGBoost Regressor, Multi-Layer Perceptron, Neural Network, or Long Short-Term Memory Network), alter the hyperparameters at the top of the notebook as you see fit, consider commenting out optional preprocessing tools (data normalization, PCA, undersampling majority class), and then run it. The wells list must match the wells that are listed in the Import and Preprocess Data Using Featmap.ipynb and Extract Labels.ipynb notebooks, so you must also re-run those notebooks every time you change which wells you want the model to consider. All of the models iterate through the wells, using one for testing and the rest for training. At the end, they will report metrics indicating how well the model performed for training and testing, as well as plot the predicted fractured depths in comparison to the true fractured depths. The trained models will be dumped into the Models folder, so you can retrieve the trained models and run it on new data without having to repeatedly retrain the model.
    * Python notebook to run: any of the notebooks in Machine Learning Models folder
      * Path: REU-Project/Python Notebooks/Machine Learning Models/
    * Where it imports from: Raw or Norm CSV Files/, labels/, Models/ 
    * Where it writes to: Models/

### Dataset Analysis
*** all in REU Project/Python Notebooks/Dataset Analysis Tools folder
  * Statistical Feature Distributions: Compare statistical distributions of each feature across wells by making a box plot of each feature. This can help identify if any wells contain outlier feature data that could hinder the model.
    * Python notebook to run: Feature Distribution Comparison.ipynb
    * Where it imports from: Raw or Norm CSV Files/
* Correlation Analysis: Plots a correlation heatmap that displays correlation magnitudes between each feature. Calculates variance inflation scores for each feature, which indicate how correlated a feature is with other features. Ideally, each VIF would be <= 5. To find an optimal set of features, employ random search algorithms that ensure all VIF scores <= n and minimizes the sum/average of VIF scores or maximizes that amount of features in the set. The user can define features that they want guaranteed to be included in the set.
    * Python notebook to run: Feature Correlation Analysis.ipynb
    * Where it imports from: Raw or Norm CSV Files/, labels/
* PCA Analysis: Scikit-Learn’s PCA analysis outputs num_features principal components that are orthogonal vectors which encode a certain amount of the dataset’s explained variance. Principal component 1 encodes the most variance, then principal component 2, and so on. It plots a scree plot that shows how much variance each principal component explains. It makes a scatter plot that shows how similar datasets are to one another. Loading scores indicate which features have a greater coefficient (and are therefore more important) in each principal component vector. The output of a PCA computation that explains n% of the data’s variance can be inputed into the ML models to help prevent overfitting.
    * Python notebook to run: PCA Analysis.ipynb
    * Where it imports from: Raw or Norm CSV Files/, labels/
* TSNE Analysis: Scikit-Learn’s TSNE analysis is a data visualization technique that projects multidimensional data into two dimensions (by optimally maintaining the original distances in 2d space), and therefore helps display clustering patterns in data. It makes various TSNE plots that can demonstrate clustering patterns in the data based on the specific well, rock formation, and fracture presence. If the data points are clearly clustered based on rock formation, for example, that indicates that the log data corresponding to the same formations across different wells is similar. If the data points do not display clustering based on the well, for example, that indicates that the log data corresponding to different wells is relatively similar. It also plots the TSNE colored by individual features, which can be used to evaluate how feature values correspond to other trends in the data.
    * Python notebook to run: TSNE Analysis.ipynb
    * Where it imports from: Raw or Norm CSV Files/, labels/

### Hyperparameter Tuning
* Implements random search and Ray Tune hyperparameter tuning to find the most optimal hyperparameters for Random Forest and XGBoost models for the current dataset. If a model was working relatively well, hyperparameter tuning could be used to better its performance.
	* Python notebook to run: Random Forest_XGBoost Hyperparameter Tuning.ipynb
   		* Path: REU-Project/Python Notebooks/Hyperparameter Tuning
