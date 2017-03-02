

# This script runs XGBoost, Lasso, and Kernel Ridge and takes the mean (blended)
# result as a final solution.
#
# I also attempted to use Keras/Tensorflow, ExtraTrees, & ElasticNet and in the end they didn't
# improve the results
#
# This script is essentially a fork and improvement (modelling, feature engineering)
# of Human Analog's excellent script posted here:
# https://www.kaggle.com/humananalog/house-prices-advanced-regression-techniques/xgboost-lasso


import numpy as np
import pandas as pd
from featuresUtil import *


# The error metric: RMSE on the log of the sale prices.
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Load the data.
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")



# There are a few houses with more than 4000 sq ft living area that are
# outliers, so we drop them from the training data. (There is also one in
# the test set but we obviously can't drop that one.)
train_df.drop(train_df[train_df["GrLivArea"] > 4000].index, inplace=True)

# The test example with ID 666 has GarageArea, GarageCars, and GarageType 
# but none of the other fields, so use the mode and median to fill them in.
test_df.loc[666, "GarageQual"] = "TA"
test_df.loc[666, "GarageCond"] = "TA"
test_df.loc[666, "GarageFinish"] = "Unf"
test_df.loc[666, "GarageYrBlt"] = "1980"

# The test example 1116 only has GarageType but no other information. We'll 
# assume it does not have a garage.
test_df.loc[1116, "GarageType"] = np.nan



# For imputing missing values: fill in missing LotFrontage values by the median
# LotFrontage of the neighborhood.
lot_frontage_by_neighborhood = train_df["LotFrontage"].groupby(train_df["Neighborhood"])



train_df_munged = munge(train_df, lot_frontage_by_neighborhood)
test_df_munged = munge(test_df, lot_frontage_by_neighborhood)

print(train_df_munged.shape)
print(test_df_munged.shape)


# Copy NeighborhoodBin into a temporary DataFrame because we want to use the
# unscaled version later on (to one-hot encode it). 
neighborhood_bin_train = pd.DataFrame(index = train_df.index)
neighborhood_bin_train["NeighborhoodBin"] = train_df_munged["NeighborhoodBin"]
neighborhood_bin_test = pd.DataFrame(index = test_df.index)
neighborhood_bin_test["NeighborhoodBin"] = test_df_munged["NeighborhoodBin"]




numeric_features = train_df_munged.dtypes[train_df_munged.dtypes != "object"].index


# Transform the skewed numeric features by taking log(feature + 1).
# This will make the features more normal.
from scipy.stats import skew

skewed = train_df_munged[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.60]
skewed = skewed.index

train_df_munged[skewed] = np.log1p(train_df_munged[skewed])
test_df_munged[skewed] = np.log1p(test_df_munged[skewed])

# Additional processing: scale the data.   
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_df_munged[numeric_features])

scaled = scaler.transform(train_df_munged[numeric_features])
for i, col in enumerate(numeric_features):
    train_df_munged[col] = scaled[:, i]

scaled = scaler.transform(test_df_munged[numeric_features])
for i, col in enumerate(numeric_features):
    test_df_munged[col] = scaled[:, i]


# Add the one-hot encoded categorical features.
onehot_df = munge_onehot(train_df)
onehot_df = onehot(onehot_df, neighborhood_bin_train, "NeighborhoodBin", None, None)
train_df_munged = train_df_munged.join(onehot_df)

# These onehot columns are missing in the test data, so drop them from the
# training data or we might overfit on them.
drop_cols = [
                "_Exterior1st_ImStucc", "_Exterior1st_Stone",
                "_Exterior2nd_Other","_HouseStyle_2.5Fin", 
            
                "_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll",
                "_Condition2_RRAe", "_Condition2_RRAn", "_Condition2_RRNn",
                "_Heating_Floor", "_Heating_OthW",

                "_Electrical_Mix", 
                "_MiscFeature_TenC",
                "_GarageQual_Ex", "_PoolQC_Fa"
            ]
train_df_munged.drop(drop_cols, axis=1, inplace=True)

onehot_df = munge_onehot(test_df)
onehot_df = onehot(onehot_df, neighborhood_bin_test, "NeighborhoodBin", None, None)
test_df_munged = test_df_munged.join(onehot_df)

# This column is missing in the training data. There is only one example with
# this value in the test set. So just drop it.
test_df_munged.drop(["_MSSubClass_150"], axis=1, inplace=True)

# Drop these columns. They are either not very helpful or they cause overfitting.
drop_cols = [
    "_Condition2_PosN",    # only two are not zero
    "_MSZoning_C (all)",
    "_MSSubClass_160",
]
train_df_munged.drop(drop_cols, axis=1, inplace=True)
test_df_munged.drop(drop_cols, axis=1, inplace=True)





# Taking the log and adding 10000 makes the housing price more normally distributed
label_df = pd.DataFrame(index = train_df_munged.index, columns=["SalePrice"])
label_df["SalePrice"] = np.log(train_df["SalePrice"] + 10000)

print("Training set size:", train_df_munged.shape)
print("Test set size:", test_df_munged.shape)




# Run XGBoost

regr2 = xgb.XGBRegressor(
                 colsample_bytree=0.6,
                 gamma=0.0,
                 #learning_rate=0.01,
                 learning_rate=0.1,
                 max_depth=3,
                 min_child_weight=3,
                 n_estimators=7000,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.005,
                 subsample=0.8,
                 seed=28,
                 objective='reg:linear',
                 silent=1)

regr2.fit(train_df_munged, label_df)

y_pred2 = regr2.predict(train_df_munged)
y_test2 = label_df
print("XGBoost score on training set: ", rmse(y_test2, y_pred2))

# Run prediction on the Kaggle test set.
y_pred_xgb2 = regr2.predict(test_df_munged)



# Run Lasso

from sklearn.linear_model import Lasso

# I found this best alpha through cross-validation.
best_alpha = 0.00098

regr = Lasso(alpha=best_alpha, max_iter=50000)
regr.fit(train_df_munged, label_df)

# Run prediction on training set to get a rough idea of how well it does.
y_pred = regr.predict(train_df_munged).ravel()
y_test = label_df
print("Lasso score on training set: ", rmse(y_test, y_pred))

# Run prediction on the Kaggle test set.
y_pred_lasso = regr.predict(test_df_munged).ravel()




# Run Extra Trees
# In the end, this wasn't used.  Didn't add to overall performance.

from sklearn.ensemble import ExtraTreesRegressor

clf = ExtraTreesRegressor(n_estimators=1000, max_depth=20, min_samples_split=20, 
                          random_state=0, max_features='auto')

clf.fit(train_df_munged, label_df.values.ravel())

y_pred = clf.predict(train_df_munged).ravel()
y_test = label_df
print("ExtraTrees score on training set: ", rmse(y_test, y_pred))

# Run prediction on the Kaggle test set.
y_pred_extraT = clf.predict(test_df_munged).ravel()



# Run Ridge & ElasticNet models.
# In the end, ElasticNet wasn't used.  Didn't add to the overall performance

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge

regr3 = ElasticNet(alpha=0.001)


regr3.fit(train_df_munged, label_df)

y_pred = regr3.predict(train_df_munged).ravel()
y_test = label_df
print("ElasticNet score on training set: ", rmse(y_test, y_pred))

# Run prediction on the Kaggle test set.
y_pred_elastic = regr3.predict(test_df_munged)


regr4 = KernelRidge(alpha=0.3, kernel='polynomial', degree=2, coef0=1.85)

regr4.fit(train_df_munged, label_df)

y_pred = regr4.predict(train_df_munged).ravel()
y_test = label_df
print("KernelRidge score on training set: ", rmse(y_test, y_pred))

# Run prediction on the Kaggle test set.
y_pred_ridge = regr4.predict(test_df_munged).ravel()





# Stack the results of the five

print "Shapes: ", y_pred_xgb.shape, y_pred_lasso.shape, y_pred_extraT.shape, y_pred_ridge.flatten().shape, y_pred_elastic.shape


#y_pred_BL = (y_pred_lasso + y_pred_xgb + y_pred_ridge.flatten() + y_pred_elastic) / 4.
y_pred_BL = (y_pred_lasso + y_pred_xgb + y_pred_ridge.flatten()) / 3.
#y_pred_BL = np.mean(y_pred_lasso + y_pred_xgb + y_pred_ridge.flatten() + y_pred_elastic, axis=0)



y_pred_xgb   = np.exp(y_pred_xgb).astype(int) - 10000
y_pred_lasso  = np.exp(y_pred_lasso).astype(int) - 10000 
y_pred_extraT = np.exp(y_pred_extraT).astype(int) - 10000
y_pred_ridge  = np.exp(y_pred_ridge.flatten()).astype(int) - 10000
y_pred_elastic= np.exp(y_pred_elastic).astype(int) - 10000
y_pred_BL     = np.exp(y_pred_BL) - 10000

y_pred_STD = np.sqrt( np.power(y_pred_lasso  - y_pred_BL , 2) +
                      np.power(y_pred_xgb   - y_pred_BL , 2) +
                      np.power(y_pred_elastic- y_pred_BL , 2) +
                      np.power(y_pred_ridge  - y_pred_BL , 2) )


print "\n\n\n"
print "blend \t xgb \t lasso \t extratree \t elastic \t ridge \n"
for i in range(0,20):
    print y_pred_BL[i], "\tvs", y_pred_BL2[i], "\t", y_pred_xgb[i], "\t", y_pred_lasso[i], "\t", y_pred_extraT[i], "\t", y_pred_elastic[i], "\t", y_pred_ridge[i], "\tstd:", y_pred_STD[i]


pred_df = pd.DataFrame(y_pred_BL, index=test_df["Id"], columns=["SalePrice"])
pred_df.to_csv('output.csv', header=True, index_label='Id')

