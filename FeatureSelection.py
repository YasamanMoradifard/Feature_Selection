# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# ------------------------------------------------------------#

# importing dataset
from sklearn.datasets import load_boston

x = load_boston()
df = pd.DataFrame(x.data, columns = x.feature_names)
df["MEDV"] = x.target

X = df.drop("MEDV", 1)   # features
Y = df["MEDV"]           # Target/observations

df.head()

# 1- Filter method ------------------------------------------------------------#

# Pearson's correlation coefficient
plt.figure(figsize=(12,12))
cor = df.corr()
sns.heatmap(cor, annot= True, cmap= plt.cm.Reds)
plt.show()

# 1- Correlation with target group
cor_target = abs(cor["MEDV"])

# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]

# 2- Correlation between selected features
print(df[["LSTAT","PTRATIO"]].corr())
print(df[["RM","LSTAT"]].corr())
print(df[["RM","PTRATIO"]].corr())


# All in a function:
def Correlation(dataset, threshold):
    X_corr = list() # all correlated features with target group
    X_selected = list() # finally selected features after evaluating correlation between X_corr features
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        if (abs(corr_matrix.iloc[i,-1]) >= threshold):
            X_corr.append(corr_matrix.columns[i]) # we have the name of features
    X_corr.remove("MEDV")
        
    return X_corr


selected_features_Filter = Correlation(df, 0.5)
print(selected_features_Filter)

# 2- Wrapper method  ------------------------------------------------------------#

# I. Backward eliminatoin #
Selected_features = list(X.columns)
P_max = 1 # P_value

while(len(Selected_features) > 0):
    P_vlaues = []
    # Adding constant column of ones
    X_1 = X[Selected_features]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(Y,X_1).fit() # fitting model on our data
    P_values = pd.Series(model.pvalues.values[1:], index = Selected_features)
    P_max = max(P_values)
    feature_P_max = P_values.idxmax()
    if(P_max > 0.05):
        Selected_features.remove(feature_P_max)
    else:
        break
        
Selected_features_BE = Selected_features


# II. Recursive Feature Elimination (RFE)
#no of features
nof_list=np.arange(1,13)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


selected_features = list(X.columns)
model = LinearRegression()
#Initializing RFE model
rfe = RFE(model, 10)             
#Transforming data using RFE
X_rfe = rfe.fit_transform(X,Y)  
#Fitting the data to model
model.fit(X_rfe,Y)              
temp = pd.Series(rfe.support_,index = selected_features)
selected_features_RFE = temp[temp==True].index
print(selected_features_RFE)

# 3- Embedded method  ------------------------------------------------------------#
# I. LASSO regularization

reg = LassoCV()
reg.fit(X, Y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,Y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using LASSO Model")