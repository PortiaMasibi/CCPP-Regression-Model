# This model was analysed in regression_models_analysis.ipynb
# If you have not taken a look at it please do to get a better idea of how we chose the regression model and optimized it
import pandas as pd 
# pip install scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
#pip install joblib
import joblib

# Importing data 
CCPP = pd.read_csv('data/CCPP.csv')

# Separating Data into X and y 
X = CCPP.drop(['electric_power'], axis = 1)
y = CCPP['electric_power']

# Split in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/3, random_state=0)

# Create the instance of the scaler
scaler = StandardScaler()

# Transform the inputs X
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Fitting the model with the best optimized parameters 
forest = RandomForestRegressor(n_estimators = 400,min_samples_split = 2, min_samples_leaf= 1)
forest.fit(X_train,y_train)
#y_pred = forest.predict(X_test)

# Saving model 
joblib.dump(forest, 'rf_model.joblib')


