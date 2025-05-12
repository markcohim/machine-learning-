##############
### Step1 ####
# EDA Analysis
##############

from pandas import read_csv
from ydata_profiling import ProfileReport
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
from matplotlib import pyplot
from sklearn.svm import SVR, NuSVR, LinearSVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, RandomizedSearchCV
from sklearn.metrics import PredictionErrorDisplay
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor
from numpy import mean, std
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline 

# file = "/Users/markco/Desktop/1212/ESG.csv"
file = "ESG.csv"
df = read_csv(file)

profile = ProfileReport(df, title= "ESG.html")
profile.to_file(output_file= "ESG.html")

###################################################################
### Step2 #########################################################
# Data Preprocessing and review data by using ydata profiling again
###################################################################

data = pd.read_csv('ESG.csv', header=0)
df = pd.DataFrame(data)

data_test = pd.read_csv('ESG_Test.csv', header=0)
df_test = pd.DataFrame(data_test)


robust_scaler = RobustScaler()
df['SG_Robust'] = robust_scaler.fit_transform(df[['SG']])
df_test['SG_Robust'] = robust_scaler.fit_transform(df_test[['SG']])


min_max_scaler = MinMaxScaler()
df['AGE_MinMax'] = min_max_scaler.fit_transform(df[['AGE']])
df_test['AGE_MinMax'] = min_max_scaler.fit_transform(df_test[['AGE']])

df = pd.get_dummies(df, columns=['IND'], prefix='IND', dtype=int)
df_test = pd.get_dummies(df_test, columns=['IND'], prefix='IND', dtype=int)


#natural log for TAT and LEV
df['LnTAT'] = np.log(df['TAT'])
df_test['LnTAT'] = np.log(df_test['TAT'])
df['LnLEV'] = np.log(df['TAT'])
df_test['LnLEV'] = np.log(df_test['TAT'])

# print(df.columns)
# print(df_test.columns)


df_numerical_columns = [
    'ID',
    'YEAR',
    'ESGS',
    'ENV',
    'SOC',
    'GOV',
    'LnTAT',
    'LnA',
    'LnLEV',
    'SG_Robust',
    'AGE_MinMax',
    'BIG4',
    'ROA',
    'ROE'
]


df_test_numerical_columns = [
    'ID',
    'YEAR',
    'ESGS',
    'ENV',
    'SOC',
    'GOV',
    'LnTAT',
    'LnA',
    'LnLEV',
    'SG_Robust',
    'AGE_MinMax',
    'BIG4'
]


scaled_df = df[df_numerical_columns]
scaled_df.to_csv("Scaled_ESG.csv", index=False)

scaled_df_test = df_test[df_test_numerical_columns]
scaled_df_test.to_csv("Scaled_ESG_Test.csv", index=False) 


file = "Scaled_ESG.csv"
df = read_csv(file)
profile = ProfileReport(df, title= "Scaled_ESG.html")
profile.to_file(output_file= "Scaled_ESG.html")


file = "Scaled_ESG_Test.csv"
df_test = read_csv(file)
profile2 = ProfileReport(df_test, title= "Scaled_ESG_Test.html")
profile2.to_file(output_file= "Scaled_ESG_Test.html")

############################################################################################################################
### Step 3 #################################################################################################################
# Model Selection : Cross Validation and Evaluation for each model ('neg_mean_absolute_error','neg_mean_squared_error','r2')
############################################################################################################################

data_test = pd.read_csv('Scaled_ESG.csv',header=0)
Scaled_df = pd.DataFrame(data_test)
print(Scaled_df)

y = Scaled_df['ROE']
X = Scaled_df.drop(columns=['ROE','ROA'], inplace=False)

# Initialise the models
Models = {
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(), 
    "Extra Trees": ExtraTreesRegressor(),
    "Extra Tree": ExtraTreeRegressor(),
    "Adaboost": AdaBoostRegressor(),
    "Bagging": BaggingRegressor(),
    "HBGT": HistGradientBoostingRegressor(), 
    "KNN": KNeighborsRegressor(), 
    "Linear SVR": LinearSVR(),
    "SVR": SVR(),
    "NuSVR" : NuSVR(),
    "MLPRegressor": MLPRegressor(),
    "KRR": KernelRidge(),
    "Dummy Regressor": DummyRegressor(),
    "Transformed Target Regressor": TransformedTargetRegressor()
}


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

evaluation_metrics = ['neg_mean_absolute_error','neg_mean_squared_error','r2']
stats_df = pd.DataFrame(columns=evaluation_metrics)

# Create a Dictionary to store cross-validation scores for each model
results = {}

# train/fit and evalute the model 
# Using cross_val_score
print("Using cross_val_score")
results_mean = []
results_std = []
for name, model in Models.items():
    #train the model
    Models = model.fit(X_train, y_train)    
    print(name)
    for evaluation_metric in evaluation_metrics:
        # print(score)
        score = cross_val_score(Models, X_train, y_train, cv=kf, scoring=evaluation_metric)
        score_mean = score.mean()
        # results_mean.append(score_mean)
        score_std = score.std()
        # results_std.append(score_std)
        print(evaluation_metric, ': ', score_mean, score_std) 
        results[name]= {"mean of MSE": score_mean, "sd of MSE": score_std}

print()


####################################################################################
### Step 4 #########################################################################
# Model Selection : 2nd Cross Validation and Evaluation for each model with plotting 
####################################################################################

# Load data 
data_test = read_csv('Scaled_ESG.csv',header=0)
Scaled_df = pd.DataFrame(data_test)
# print(Scaled_df)

y = Scaled_df['ROE']
X = Scaled_df.drop(columns=['ROE','ROA'], inplace=False)

# Define models
Models = [
    DecisionTreeRegressor(),
    RandomForestRegressor(), 
    GradientBoostingRegressor(),
    ExtraTreesRegressor(),
    ExtraTreeRegressor(),
    AdaBoostRegressor(),
    BaggingRegressor(),
    HistGradientBoostingRegressor(), 
    KNeighborsRegressor(), 
    SVR(),
    NuSVR(),
    KernelRidge(),
    DummyRegressor(),
    TransformedTargetRegressor(),
    LGBMRegressor(),
    XGBRegressor()
    ]


Model_list =[]
for Model in Models:
    Model_list += [Model.__class__.__name__]
    
# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf=5


# Using cross_val_score
print("2. Using cross_val_score")

evaluation_metrics = ['neg_mean_squared_error']

# Create a DataFrame to store cross-validation scores for each model
stats_score_df = pd.DataFrame(columns=Model_list)

results_mean = []
results_std = []
results = {}


for Model in Models:
    for evaluation_metric in evaluation_metrics:
        score = cross_val_score(Model, X, y, cv=kf, scoring=evaluation_metric)
        # print(Model, score)
        results[Model.__class__.__name__] = score
        # print(-score)
        score_mean = float(score.mean())
        results_mean.append(score_mean)
        score_std = float(score.std())
        results_std.append(score_std)


stats_score_dict = {
'Model': Model_list,
'Mean': results_mean,
'Std': results_std
}

# Set display format for floats
pd.options.display.float_format = '{:,.6f}'.format
# print(stats_score_dict)
stats_score_df = pd.DataFrame(stats_score_dict)
print(stats_score_df)

# Convert the dictionary to a DataFrame
results_df = pd.DataFrame(results)

# Plotting boxplot
pyplot.figure(figsize=(10, 6))
results_df.boxplot()
pyplot.title('Cross-Validation Scores for Different Regressors')
pyplot.ylabel('neg_mean_squared_error')
pyplot.xticks(rotation=90)
pyplot.grid(axis='y')
pyplot.show()



#########################################################################################################
### Step 5 ##############################################################################################
# Model Selection : Cross Validation and Evaluaiton for each model with plotting (combined with stacking)
#########################################################################################################


def get_stacking():
    # define the base models
    level0 = list()
    # level0.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
    level0.append(('RandomForestRegressor', RandomForestRegressor()))
    level0.append(('GradientBoostingRegressor', GradientBoostingRegressor()))
    # level0.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
    # level0.append(('ExtraTreeRegressor', ExtraTreeRegressor()))
    level0.append(('AdaBoost', AdaBoostRegressor()))	
    level0.append(('BaggingRegressor', BaggingRegressor()))
    level0.append(('HistGradientBoostingRegressor', HistGradientBoostingRegressor()))
    level0.append(('KNN', KNeighborsRegressor()))
    # level0.append(('SVR', SVR()))
    # level0.append(('NuSVR', NuSVR()))
    # level0.append(('KRR', KernelRidge()))   
    # level0.append(('DummyRegressor', DummyRegressor()))   
    # level0.append(('TransformedTargetRegressor', TransformedTargetRegressor()))   
    # level0.append(('LightGBM', LGBMRegressor()))
    # level0.append(('xgBoost', XGBRegressor()))
    # level1 = GradientBoostingRegressor()
    level1 = ExtraTreesRegressor()
    # level1 = HistGradientBoostingRegressor()
    # level1 = LGBMRegressor()
    # level1 = XGBRegressor()
	# define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model
         

# get a list of models to evaluate
def get_models():
    models = dict()
    models['DecisionTreeRegressor'] = DecisionTreeRegressor()
    models['RandomForestRegressor'] = RandomForestRegressor()
    models['GradientBoostingRegressor'] = GradientBoostingRegressor()
    models['ExtraTreesRegressor'] = ExtraTreesRegressor()
    models['ExtraTreeRegressor'] = ExtraTreeRegressor()
    models['AdaBoost'] = AdaBoostRegressor()
    models['BaggingRegressor'] = BaggingRegressor()
    models['HistGradientBoostingRegressor'] = HistGradientBoostingRegressor()
    models['KNN'] = KNeighborsRegressor()
    models['SVR'] = SVR()
    models['NuSVR'] = NuSVR()
    models['KRR'] = KernelRidge()
    models['DummyRegressor'] = DummyRegressor()
    models['TransformedTargetRegressor'] = TransformedTargetRegressor()
    models['LightGBM'] = LGBMRegressor()
    models['xgBoost'] = XGBRegressor()
    models['Stacking'] = get_stacking()
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=kf, scoring=evaluation_metric)
    return scores

# Define the error plot function
def Error_Plot(Model, y, y_pred):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=None,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values " + Model)
    PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=None,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    plt.tight_layout()
    plt.show()


# define dataset
# X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    # print(model, scores)    
    results.append(scores)
    names.append(name)
    # print(name, mean(scores),std(scores))
    print('>%s %.6f (%.6f)' % (name, mean(scores),std(scores)))
    stacking_y_pred = cross_val_predict(model, X, y, cv=kf)
    Error_Plot(name, y, stacking_y_pred)


# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.xticks(rotation=90)
pyplot.title('Cross-Validation Scores for Different Regressors')
pyplot.ylabel('neg_mean_squared_error')
pyplot.show()



#############################################
### Step 6 ##################################
# Cross Validation and Stacking with Pipeline
#############################################

# Define level 0 models (base learners) 
level_0_models = [
    ('rf', RandomForestRegressor()),  # Changed names to shorter versions
    ('gb', GradientBoostingRegressor()),
    ('ada', AdaBoostRegressor()),
    ('bag', BaggingRegressor()),
    ('hist', HistGradientBoostingRegressor()),
    ('knn', KNeighborsRegressor())
]

# Define the meta-model (level 1 model) 
meta_model = ExtraTreesRegressor() 

# Create a StackingRegressor 
stacking_model = StackingRegressor(     
    estimators=level_0_models,     
    final_estimator=meta_model 
)

# Set up the parameter grid for hyperparameter tuning 
param_grid = {     
    'stackingregressor__final_estimator__n_estimators': [50, 100],  # For ExtraTreesRegressor
    'stackingregressor__final_estimator__max_depth': [None, 10, 20],  # For ExtraTreesRegressor
    'stackingregressor__rf__n_estimators': [50, 100],  # For RandomForestRegressor
    'stackingregressor__rf__max_depth': [None, 10, 20],  # For RandomForestRegressor
    'stackingregressor__gb__n_estimators': [50, 100],  # For GradientBoostingRegressor
    'stackingregressor__gb__learning_rate': [0.01, 0.1, 0.2],  # For GradientBoostingRegressor
    'stackingregressor__ada__n_estimators': [50, 100],  # For AdaBoostRegressor
    'stackingregressor__bag__n_estimators': [50, 100],  # For BaggingRegressor
    'stackingregressor__hist__max_iter': [100, 200],  # For HistGradientBoostingRegressor
    'stackingregressor__knn__n_neighbors': [3, 5, 7]  # For KNeighborsRegressor
}

# Create a pipeline with scaling and stacking model 
pipeline = Pipeline([
    # ('scaler', StandardScaler()),               # Task 1: Scaling - Standardize features     
    ('stackingregressor', stacking_model)      # Task 2: Model Training - StackingRegressor
])

# Set up cross-validation 
kf = KFold(n_splits=5)

# Set up RandomizedSearchCV 
grid = RandomizedSearchCV(pipeline, param_grid, cv=kf, scoring='neg_mean_squared_error')  # Using neg_mean_squared_error for regression

# Fit RandomizedSearchCV
grid.fit(X, y)

# Best parameters and score 
print("Best parameters:", grid.best_params_)

# Access the results 
results = pd.DataFrame(grid.cv_results_)

# Save the results to a CSV file for manual inspection 
results.to_csv('RandomizedSearchCV_results_code_2.csv', index=False)

# Optionally, display the results DataFrame 
print("Best cross-validation score:", grid.best_score_)


# Load the test data
data_test = pd.read_csv('Scaled_ESG_Test.csv', header=0)
Scaled_df_Test = pd.DataFrame(data_test)

# Fit the stacking model to the training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
stacking_model.fit(X_train, y_train)

#####################################################################
### Step 7 ##########################################################
# Prediction and Sensitivity Analysis with trained models by Pipeline
#####################################################################

# Make predictions on the test data
# Ensure that you are using only the original features
y_test_pred = stacking_model.predict(Scaled_df_Test.drop(columns=['y_test_pred'], errors='ignore'))  # Drop 'y_test_pred' if it exists

# Print the predictions
print(y_test_pred)

# Create a new DataFrame for predictions
predictions_df = pd.DataFrame({'y_test_pred': y_test_pred})

# Concatenate predictions with the original test DataFrame (without modifying the original)
result_df = pd.concat([Scaled_df_Test, predictions_df], axis=1)

# Save the updated DataFrame to a new CSV file
result_df.to_csv('MTT_y_predict.csv', index=False)

# Sensitivity Analysis
sensitivity_results = {}

# Select variables to analyze (e.g., 'SG_Robust' and 'AGE_MinMax')
variables_to_analyze = ['SG_Robust', 
                        'AGE_MinMax', 
                        'ESGS',
                        'LnTAT',
                        'LnA',
                        'ENV',
                        'SOC',
                        'GOV',
                        'LnLEV']

# Define the percentage changes to test
percentage_changes = np.arange(-0.5, 0.6, 0.1)  # From -50% to +50%

for variable in variables_to_analyze:
    original_values = result_df[variable].copy()  # Use result_df instead of Scaled_df_Test
    for change in percentage_changes:
        # Create a modified dataset
        modified_X_test = result_df.copy()
        
        # Apply the change
        modified_X_test[variable] = original_values * (1 + change)
        
        # Predict with the modified dataset
        y_modified_pred = stacking_model.predict(modified_X_test.drop(columns=['y_test_pred'], errors='ignore'))  # Drop 'y_test_pred'
        
        # Calculate the change in predictions
        sensitivity_results[(variable, change)] = y_modified_pred.mean() - y_test_pred.mean()

# Convert results to DataFrame for easier plotting
sensitivity_df = pd.DataFrame(list(sensitivity_results.items()), columns=['Variable_Change', 'Change in Prediction'])
sensitivity_df['Variable'] = sensitivity_df['Variable_Change'].apply(lambda x: x[0])
sensitivity_df['Percentage Change'] = sensitivity_df['Variable_Change'].apply(lambda x: x[1] * 100)  # Convert to percentage
sensitivity_df['Change in Prediction'] = sensitivity_df['Change in Prediction']

# Plotting the sensitivity analysis results
plt.figure(figsize=(12, 6))
for variable in variables_to_analyze:
    subset = sensitivity_df[sensitivity_df['Variable'] == variable]
    plt.plot(subset['Percentage Change'], subset['Change in Prediction'], marker='o', label=variable)

plt.title('Sensitivity Analysis of Variables')
plt.xlabel('Percentage Change in Variables')
plt.ylabel('Change in Predicted ROE')
plt.axhline(0, color='grey', linestyle='--', linewidth=0.7)
plt.legend()
plt.grid()
plt.show()

# Print the results
print(sensitivity_df)





