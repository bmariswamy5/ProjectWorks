#%%[markdown]
## Team T8

## Imports ##
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.linear_model import LogisticRegression
from statsmodels.formula.api import ols
from statsmodels.formula.api import glm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error



#%%
####################### EDA ##########################
## Load data into pd dataframe
bar_df=pd.read_csv('bar_pass_prediction.csv')

print(f'Number of observarions in df...:{len(bar_df)}')
print(f'DF column count: {len(bar_df.columns)}')
print(f'Show shape of the df...:{(bar_df.shape)}')

#%%
print('\nShow the  information about the df...')
bar_df.info()
print('\nDescribe the  information about the df...')
bar_df.describe()
print('\ndisplay head of df...\n')
bar_df.head()


# %%
## Check for duplicates and Null values in the df
Uniqueval = bar_df['ID'].nunique()
Totalval = bar_df.shape[0]
duplicates= Totalval - Uniqueval
print('Number of duplicate in the df:', duplicates)
# We observe that there are no duplicate values in the df.
bar_df.isnull().sum()
# We observe that they are few Null values in df.

# %%
### Dropping Null values and variables which is not required for analysis
# We filtered out variables not used in our analyses.
df=bar_df.drop(['decile1b','male','decile3','ID','decile1','cluster','zfygpa','DOB_yr','zgpa','bar1','bar1_yr','bar2','bar2_yr','Dropout','bar','bar_passed','index6040','indxgrp','indxgrp2','dnn_bar_pass_prediction','race','race2','other','asian','black','hisp', 'parttime','grad','sex', 'age'], axis=1)
# We dropped Null values in df.
df = df.dropna()
df.isnull().sum()

#%%
df.info()
print(f'DF column count: {len(df.columns)}')
df.describe()

#%%
# We converted the nominal categorical to numeric variables
df['race1'].replace({'white':1, 'black':2, 'hisp':3, 'asian':4, 'other':5}, inplace=True)
df['gender'].replace({'female':0, 'male':1}, inplace=True)
df['fulltime'].replace({1:1, 2:0}, inplace=True)
#print(df)

# We converted the float to int 
df['fulltime'] = df['fulltime'].astype(int)
# df['fulltime'].value_counts()
df['tier'] = df['tier'].astype(int)
# df['tier'].value_counts()
#%%
##PIE Chart
##calculating total percenatge of male/female candidates who appeared for bar exam
def gender_fun(val):
    return f'{val/100*len(df):.0f}\n{val:.0f}%'
fig,ax=plt.subplots(ncols=1,figsize=(10,5))    
df.groupby('gender').size().plot(kind='pie',autopct=gender_fun,textprops={'fontsize':11},colors=['pink','lightblue'],ax=ax)
plt.title('Pie chart distribution of Male and Female who appeared for bar')
plt.show()

def race_fun(val2):
    return f'{val2/100*len(df):.0f}\n{val2:.0f}%'
fig,ax1=plt.subplots(ncols=1,figsize=(20,10))    
df.groupby('race1').size().plot(kind='pie',autopct=race_fun,textprops={'fontsize':11},colors=['pink','lightblue','grey','orange','lightgreen','red'],ax=ax1)
plt.title('Pie chart distribution of Male and Female who appeared for bar')
plt.show()

##Total percentage of students who passed the bar
plt.figure(figsize=(6,6))
df['pass_bar'].value_counts().plot.pie( explode = [.1,.1],autopct='%1.1f%%', fontsize=14,colors=['lightblue','grey']).set_title('Percentage of students Passed Bar')

#Total percentage of students who enrolled for full time course
plt.figure(figsize=(6,6))
df['fulltime'].value_counts().plot.pie( explode = [.1,.1],autopct='%1.1f%%', fontsize=14,colors=['yellow','grey']).set_title('Percenatge of students enrolled for fulltime course')

##Total percentage of students who enrolled in different tier university
def tier_fun(val1):
    return f'{val1/100*len(df):.0f}\n{val1:.0f}%'
fig,ax=plt.subplots(ncols=1,figsize=(25,10))    
df.groupby('tier').size().plot(kind='pie',autopct=tier_fun,textprops={'fontsize':15},colors=['Pink','purple','grey','yellow','green','orange'],ax=ax)
plt.title('enrolled tier university')
plt.show()


#%%
## Data Description
df.info()
round(df.describe(), 2)
# %%
### Normality Check (Histogram) for LSAT, GPA, Undergraduate GPA, and Family Income

plt.hist(df['lsat'], label='lsat')
plt.xlabel("lsat score")
plt.ylabel("lsat Disribution")
plt.title("Distribution of lsat score")
plt.show()

plt.hist(df['gpa'], label='gpa')
plt.xlabel("gpa score")
plt.ylabel("gpa Disribution")
plt.title("Distribution of gpa")
plt.show()

plt.hist(df['ugpa'], label='ugpa')
plt.xlabel("ugpa score")
plt.ylabel("ugpa Disribution")
plt.title("Distribution of ugpa")
plt.show()
#plt.hist(df['fam_inc'], label='fam_inc')
#plt.show()

# %%
### Multiple Boxplot & ANOVA Test
## LSAT Score
# 1. LSAT Score by University Tier
# Boxplot
df['tier'].value_counts()
dftier1 = df[df['tier']==1 ]
dftier2 = df[df['tier']==2 ]
dftier3 = df[df['tier']==3 ]
dftier4 = df[df['tier']==4 ]
dftier5 = df[df['tier']==5 ]
dftier6 = df[df['tier']==6 ]
fig, ax = plt.subplots()
data = [dftier1['lsat'], dftier2['lsat'], dftier3['lsat'], dftier4['lsat'], dftier5['lsat'], dftier6['lsat']]
plt.boxplot(data)
plt.xlabel('University Tier')
plt.ylabel('LSAT Score')
# ANOVA
model = ols('lsat ~ C(tier)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 2. LSAT Score by Family Income
# Boxplot 
dfincome1 = df[df['fam_inc']==1 ]
dfincome2 = df[df['fam_inc']==2 ]
dfincome3 = df[df['fam_inc']==3 ]
dfincome4 = df[df['fam_inc']==4 ]
dfincome5 = df[df['fam_inc']==5 ]
fig, ax = plt.subplots()
data = [dfincome1['lsat'], dfincome2['lsat'], dfincome3['lsat'], dfincome4['lsat'], dfincome5['lsat']]
plt.boxplot(data)
plt.xlabel('Family Income')
plt.ylabel('LSAT Score')
# ANOVA
model = ols('lsat ~ C(fam_inc)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 3. LSAT Score by Race
# Boxplot 
dfrace1 = df[df['race1']==1 ]
dfrace2 = df[df['race1']==2 ]
dfrace3 = df[df['race1']==3 ]
dfrace4 = df[df['race1']==4 ]
dfrace5 = df[df['race1']==5 ]
fig, ax = plt.subplots()
data = [dfrace1['lsat'], dfrace2['lsat'], dfrace3['lsat'], dfrace4['lsat'], dfrace5['lsat']]
plt.boxplot(data)
plt.xlabel('Race Group')
plt.ylabel('LSAT Score')
# ANOVA
model = ols('lsat ~ C(race1)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 4.LSAT Score by Gender
# Boxplot 
dfgender1 = df[df['gender']==0 ]
dfgender2 = df[df['gender']==1 ]
fig, ax = plt.subplots()
data = [dfgender1['lsat'], dfgender2['lsat']]
plt.boxplot(data)
plt.xlabel('Gender')
plt.ylabel('LSAT Score')
# ANOVA
model = ols('lsat ~ C(gender)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

#%%
## GPA
# 1. GPA by University Tier
# Boxplot
fig, ax = plt.subplots()
data = [dftier1['gpa'], dftier2['gpa'], dftier3['gpa'], dftier4['gpa'], dftier5['gpa'], dftier6['gpa']]
plt.boxplot(data)
plt.xlabel('University Tier')
plt.ylabel('GPA')
# ANOVA
model = ols('gpa ~ C(tier)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 2. GPA by Family Income
# Boxplot
fig, ax = plt.subplots()
data = [dfincome1['gpa'], dfincome2['gpa'], dfincome3['gpa'], dfincome4['gpa'], dfincome5['gpa']]
plt.boxplot(data)
plt.xlabel('Family Income')
plt.ylabel('GPA')
# ANOVA
model = ols('gpa ~ C(fam_inc)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 3. GPA by Race
# Boxplot
fig, ax = plt.subplots()
data = [dfrace1['gpa'], dfrace2['gpa'], dfrace3['gpa'], dfrace4['gpa'], dfrace5['gpa']]
plt.boxplot(data)
plt.xlabel('Race Group')
plt.ylabel('GPA')
# ANOVA
model = ols('gpa ~ C(race1)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 4. GPA by Gender
# Boxplot
fig, ax = plt.subplots()
data = [dfgender1['gpa'], dfgender2['gpa']]
plt.boxplot(data)
plt.xlabel('Gender')
plt.ylabel('GPA')
# ANOVA
model = ols('gpa ~ C(gender)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

#%%
## Undergraduate GPA
# 1. Undergraduate GPA by University Tier
# Boxplot 
fig, ax = plt.subplots()
data = [dftier1['ugpa'], dftier2['ugpa'], dftier3['ugpa'], dftier4['ugpa'], dftier5['ugpa'], dftier6['ugpa']]
plt.boxplot(data)
plt.xlabel('University Tier')
plt.ylabel('Undergraduate GPA')
# ANOVA
model = ols('ugpa ~ C(tier)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 2. Undergraduate GPA by Family Income
# Boxplot 
fig, ax = plt.subplots()
data = [dfincome1['ugpa'], dfincome2['ugpa'], dfincome3['ugpa'], dfincome4['ugpa'], dfincome5['ugpa']]
plt.boxplot(data)
plt.xlabel('Family Income')
plt.ylabel('Undergraduate GPA')
# ANOVA
model = ols('ugpa ~ C(fam_inc)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 3. Undergraduate GPA by Race
# Boxplot 
fig, ax = plt.subplots()
data = [dfrace1['ugpa'], dfrace2['ugpa'], dfrace3['ugpa'], dfrace4['ugpa'], dfrace5['ugpa']]
plt.boxplot(data)
plt.xlabel('Race Group')
plt.ylabel('Undergraduate GPA')
# ANOVA
model = ols('ugpa ~ C(race1)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 4. Undergraduate GPA by Gender
# Boxplot 
fig, ax = plt.subplots()
data = [dfgender1['ugpa'], dfgender2['ugpa']]
plt.boxplot(data)
plt.xlabel('Gender')
plt.ylabel('Undergraduate GPA')
# ANOVA
model = ols('ugpa ~ C(gender)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)


# %%
### Correlation Table
# Type 1
corrmatrix = df.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix, vmax=.3, square=True)

# Type 2
from pandas.plotting import scatter_matrix
# scatter_matrix(xpizza, alpha = 0.2, figsize = (7, 7), diagonal = 'hist')
scatter_matrix(df, alpha = 0.2, figsize = (7, 7), diagonal = 'kde')
# plt.title("pandas scatter matrix plot")
plt.show()

# Type 3
import seaborn as sns
# sns.set()
sns.pairplot(df)
plt.title("seaborn pairplot")
plt.show()


# %%
## Simple Statistic Analyses for Linear Regression & Logit Regression
from statsmodels.formula.api import ols
from statsmodels.formula.api import glm
import statsmodels.api as sm 

## Linear Regression
ols1 = ols(formula='lsat ~ ugpa+fulltime+fam_inc+gender+C(race1)+C(tier)+gpa', data=df).fit()
ols1.summary()


#%%
#ANOVA
from statsmodels.stats.anova import anova_lm
print(f"Overall model F({ols1.df_model: .0f},{ols1.df_resid: .0f}) = {ols1.fvalue: .3f}, p = {ols1.f_pvalue: .4f}")
aov_table = anova_lm(ols1, typ=2)
print(aov_table.round(4))

#Check the Normal distribution of residuals

#Method 1: Shapiro Wilk test
from scipy import stats
w, pvalue = stats.shapiro(ols1.resid)
print(w, pvalue)

#Method 2: Q-Q plot test
result = ols1.resid
fig = sm.qqplot(result, line='s')
plt.show()


#%%
## Preprocessing for ML models for LM, Decision Regression Tree, PCA
## Subset df into X and y
X = df[['ugpa', 'fulltime', 'fam_inc', 'gender', 'race1', 'tier', 'gpa']]
y = df['lsat']

X.shape
print(X.head())
print(type(X))

y.shape
print(y.head())
print(type(y))

#%%
#Linear regression and Logistic Regression using sklearn
import numpy as np
from sklearn import linear_model

## Linear Regression
from pandas.plotting import scatter_matrix
scatter_matrix(X, alpha = 0.2, figsize = (7, 7), diagonal = 'hist')
# plt.title("pandas scatter matrix plot")
plt.show()

import seaborn as sns
# sns.set()
sns.pairplot(X)
plt.title("seaborn pairplot")
plt.show()

#%%
Lsatfit = linear_model.LinearRegression()  
Lsatfit.fit(X , y)
print('score:', Lsatfit.score(X, y))
print('intercept:', Lsatfit.intercept_)  
print('coef_:', Lsatfit.coef_) 

#%%
# Model evaluation using train and test sets 
from sklearn.model_selection import train_test_split

# Split data into 80% train and 20% test
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.20, random_state=1)
full_split1 = linear_model.LinearRegression() # new instancew
full_split1.fit(X_train1, y_train1)
y_pred1 = full_split1.predict(X_test1)
full_split1.score(X_test1, y_test1)

print('x_trainpass shape',X_train1.shape)
print('x_testpass shape',X_test1.shape)
print('score (train):', full_split1.score(X_train1, y_train1)) 
print('score (test):', full_split1.score(X_test1, y_test1)) 
print('intercept:', full_split1.intercept_) 
print('coef_:', full_split1.coef_)  

#%% Cross validation for linear regression model
from sklearn.model_selection import cross_val_score
full_cv = linear_model.LinearRegression()
cv_results = cross_val_score(full_cv, X, y, cv=5)
print(cv_results) 
print(np.mean(cv_results))


#%%
## Decision Regression Trees 
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import mean_squared_error as MSE  # Import mean_squared_error as MSE
# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2,random_state=1)

# Instantiate a DecisionTreeRegressor 'regtree0'
regtree0 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1,random_state=22)
regtree0.fit(X_train, y_train)  # Fit regtree0 to the training set

# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# evaluation
y_test_pred = regtree0.predict(X_test)  # Compute y_test_pred
mse_regtree0 = MSE(y_test, y_test_pred)  # Compute mse_regtree0
rmse_regtree0 = mse_regtree0 ** (.5) # Compute rmse_regtree0
print("Test set RMSE of regtree0: {:.2f}".format(rmse_regtree0))

# Compare the performance with OLS
from sklearn import linear_model
ols2 = linear_model.LinearRegression() 
ols2.fit(X_train, y_train)

y_pred_ols = ols2.predict(X_test)  # Predict test set labels/values
mse_ols = MSE(y_test, y_pred_ols)  # Compute mse_ols
rmse_ols = mse_ols**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_ols))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_regtree0))

# Graphing the tree 1
from sklearn.tree import plot_tree
plt.figure(figsize=(10,8), dpi=150)
plot_tree(regtree0, feature_names=X.columns);

# Graphing the tree 2 (Result: Same as 1)
# pip install graphviz
from sklearn.tree import export_graphviz 
import os
print(os.getcwd())
# export the decision tree to a tree.dot file to visualize the plot 
export_graphviz(regtree0, out_file = 'tree.dot' , feature_names =['ugpa', 'fulltime', 'fam_inc', 'gender', 'race1', 'tier', 'gpa']) 
# Generate the Graph with .dot file at http://www.webgraphviz.com/

# %%
# PCR
# scaling the data for a better fit in PCR
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Define cross-validation folds
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Linear Regression
lin_reg = LinearRegression().fit(X_train_scaled, y_train)
lr_score_train = -1 * cross_val_score(lin_reg, X_train_scaled, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
lr_score_test = mean_squared_error(y_test, lin_reg.predict(X_test_scaled), squared=False)

# Lasso Regression
lasso_reg = LassoCV().fit(X_train_scaled, y_train)
lasso_score_train = -1 * cross_val_score(lasso_reg, X_train_scaled, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
lasso_score_test = mean_squared_error(y_test, lasso_reg.predict(X_test_scaled), squared=False)

# Ridge Regression
ridge_reg = RidgeCV().fit(X_train_scaled, y_train)
ridge_score_train = -1 * cross_val_score(ridge_reg, X_train_scaled, y_train, cv=cv, scoring='neg_root_mean_squared_error').mean()
ridge_score_test = mean_squared_error(y_test, ridge_reg.predict(X_test_scaled), squared=False)


# Generate all the principal components
pca = PCA()
X_train_pc = pca.fit_transform(X_train_scaled)

# View first 5 rows of all principal components
pd.DataFrame(pca.components_.T).loc[:4,:]

# Initialize linear regression instance
lin_reg = LinearRegression()

# Create empty list to store RMSE for each iteration
rmse_list = []

# Loop through different count of principal components for linear regression
for i in range(1, X_train_pc.shape[1]+1):
    rmse_score = -1 * cross_val_score(lin_reg, 
                                      X_train_pc[:,:i], # Use first k principal components
                                      y_train, 
                                      cv=cv, 
                                      scoring='neg_root_mean_squared_error').mean()
    rmse_list.append(rmse_score)
    
# Visual analysis - plot RMSE vs count of principal components used
plt.plot(rmse_list, '-o')
plt.xlabel('Number of principal components in Linear regression')
plt.ylabel('RMSE')
plt.title('Passed the bar?')
plt.xlim(xmin=-1);
plt.xticks(np.arange(X_train_pc.shape[1]), np.arange(1, X_train_pc.shape[1]+1))
plt.axhline(y=lr_score_train, color='g', linestyle='-');

# Visually determine optimal number of principal components
best_pc_num = 6

# Train model with first 6 principal components
lin_reg_pc = LinearRegression().fit(X_train_pc[:,:best_pc_num], y_train)

# Get cross-validation RMSE (train set)
pcr_score_train = -1 * cross_val_score(lin_reg_pc, 
                                       X_train_pc[:,:best_pc_num], 
                                       y_train, 
                                       cv=cv, 
                                       scoring='neg_root_mean_squared_error').mean()

# Train model on training set
lin_reg_pc = LinearRegression().fit(X_train_pc[:,:best_pc_num], y_train)

# Get first 6 principal components of test set
X_test_pc = pca.transform(X_test_scaled)[:,:best_pc_num]

# Predict on test data
preds = lin_reg_pc.predict(X_test_pc)
pcr_score_test = mean_squared_error(y_test, preds, squared=False)
print(f"Using PCR analysis, we found out that the first 6 variables are most optimal for training the regression models and it wouldn't make much of a difference even if we didn't include the 7th variable. We were able to reduce the RMSE to {pcr_score_test}")


#%%
## Statistical Logit Regression
logit1 = glm(formula='pass_bar ~ ugpa+fulltime+fam_inc+gender+race1+C(tier)+gpa+lsat', families = sm.families.Binomial(), data=df).fit()
logit1.summary()

###############################################################################################
#%%
## Preprocessing for ML models for Logit and KNN
## Subset df into X and y
X = df[['ugpa', 'fulltime', 'fam_inc', 'gender', 'race1', 'tier', 'gpa', 'lsat']]
y = df['pass_bar']

#%%
## Logistic Regression before SMOTE
from sklearn.linear_model import LogisticRegression

## Subset df into X and y
Xpassbar=df[['ugpa','fulltime','fam_inc','gender','race1','tier','gpa','lsat']]
print(Xpassbar.head())
print(type(Xpassbar))
Ypassbar=df['pass_bar']
print(Ypassbar.head())
print(type(Ypassbar))

#%%
# Split data into train and test set
x_trainpass, x_testpass, y_trainpass, y_testpass= train_test_split(Xpassbar, Ypassbar, random_state=1 )
print('x_trainpass type',type(x_trainpass))
print('x_trainpass shape',x_trainpass .shape)

print('x_testpass type',type(x_testpass))
print('x_testpass shape',x_testpass.shape)

print('y_trainpass type',type(y_trainpass))
print('y_trainpass shape',y_trainpass.shape)

print('y_testpass type',type(y_testpass))
print('y_testpass shape',y_testpass.shape)

#%%
# Accuracy of Logit
Barpasslogit = LogisticRegression() 
Barpasslogit.fit(Xpassbar, Ypassbar)
print('Logit model accuracy (with the test set):', Barpasslogit.score(x_testpass, y_testpass))
print('Logit model accuracy (with the train set):', Barpasslogit.score(x_trainpass, y_trainpass))

# Classification Report for logistic Regresssion
from sklearn.metrics import classification_report
y_true, y_pred = y_testpass, Barpasslogit.predict(x_testpass)
print(classification_report(y_true, y_pred))


#%%
## Receiver Operator Characteristics (ROC)
## Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_testpass))]
# predict probabilities
lr_probs = Barpasslogit.predict_proba(x_testpass)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_testpass, ns_probs)
lr_auc = roc_auc_score(y_testpass, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_testpass, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_testpass, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#%%
## Precision-Recall vs Threshold
y_pred=Barpasslogit.predict(x_testpass)
y_pred_probs=Barpasslogit.predict_proba(x_testpass) 
  # probs_y is a 2-D array of probability of being labeled as 0 (first 
  # column of array) vs 1 (2nd column in array)

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_testpass, y_pred_probs[:, 1]) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

print("\nReady to continue.")

#%% 
## Conclusion: Good with 0.8 Threshold
cut_off = 0.8
predictions = (Barpasslogit.predict_proba(x_testpass)[:,1]>cut_off).astype(int)
print(predictions)

##########################################################################################################
#%%
#Balancing the dataset for logistic regression
## Logistic Regression after SMOTE
import imblearn
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 333)
X_train_res, y_train_res = sm.fit_resample(x_trainpass, y_trainpass.ravel())

#%%
# Accuracy of Logit
Barpasslogit = LogisticRegression() 
Barpasslogit.fit(X_train_res,y_train_res)
print('Logit model accuracy (with the test set):', Barpasslogit.score(x_testpass, y_testpass))
print('Logit model accuracy (with the train set):', Barpasslogit.score(x_trainpass, y_trainpass))

# Classification Report for logistic Regresssion
from sklearn.metrics import classification_report
y_true, y_pred = y_testpass, Barpasslogit.predict(x_testpass)
print(classification_report(y_true, y_pred))


#%%
## Receiver Operator Characteristics (ROC)
## Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_testpass))]
# predict probabilities
lr_probs = Barpasslogit.predict_proba(x_testpass)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_testpass, ns_probs)
lr_auc = roc_auc_score(y_testpass, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_testpass, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_testpass, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


#%%
## Precision-Recall vs Threshold
y_pred=Barpasslogit.predict(x_testpass)
y_pred_probs=Barpasslogit.predict_proba(x_testpass) 
  # probs_y is a 2-D array of probability of being labeled as 0 (first 
  # column of array) vs 1 (2nd column in array)

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_testpass, y_pred_probs[:, 1]) 
   #retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[: -1], "b--", label="Precision")
plt.plot(thresholds, recall[: -1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0,1])

print("\nReady to continue.")

#%% 
## Conclusion: Good with 0.8 Threshold
cut_off = 0.2
predictions = (Barpasslogit.predict_proba(x_testpass)[:,1]>cut_off).astype(int)
print(predictions)

#######################################################################################################
#%%
## KNN before SMOTE
## KNN
KNN = 9
knn_cv = KNeighborsClassifier(n_neighbors=KNN) 
cv_results = cross_val_score(knn_cv, X, y, cv=10)
print(cv_results) 
print(np.mean(cv_results)) 

#%%
# scale the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('pass_bar', axis = 1))
scaled_features = scaler.transform(df.drop('pass_bar', axis = 1))

# Split data into 80% train and 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['pass_bar'], test_size = 0.20)
#X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['pass_bar'], random_state = 1)


# trying to come up with a model to predict whether someone will TARGET CLASS or not.
# start with k = 1.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations
from sklearn.metrics import classification_report, confusion_matrix
print('WITH K = 1')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

#%%
# Plot Error Rate vs. k Value
error_rate = []
for i in range(1, 40):     
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
 
plt.figure(figsize =(10, 6))
plt.plot(range(1, 40), error_rate, color ='blue',
                linestyle ='dashed', marker ='o',
         markerfacecolor ='red', markersize = 10)
 
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#%%
# now with k = 11
knn = KNeighborsClassifier(n_neighbors = 11)
 
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
 
print('WITH K = 11')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


#%%
# KNN after SMOTE

# scale the dataset

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('pass_bar', axis = 1))
scaled_features = scaler.transform(df.drop('pass_bar', axis = 1))

# Split data into 80% train and 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['pass_bar'], test_size = 0.20)
#X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['pass_bar'], random_state = 1)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)


#%%
# Plot Error Rate vs. k Value
error_rate = []
for i in range(1, 40):     
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
 
plt.figure(figsize =(10, 6))
plt.plot(range(1, 40), error_rate, color ='blue',
                linestyle ='dashed', marker ='o',
         markerfacecolor ='red', markersize = 10)
 
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

#%%

knn = KNeighborsClassifier(n_neighbors=9)
model=knn.fit(X_train, y_train)
pred = model.predict(X_test)
pred
print("Training set score: {:.2f}".format(knn.score(X_train, y_train)))
print("Validation set score: {:.2f}".format(knn.score(X_test, y_test)))
print('WITH K = 11')
print('\n')
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

# KNN Plot
#%pip install mglearn
# in case eoor occurs in this package please intall %pip install joblib==1.1.0
import mglearn
mglearn.plots.plot_knn_classification(n_neighbors=11)
plt.show()

###################################################################################################################

# %%
## KMeans 
from sklearn.cluster import KMeans

# Plot for GPA & LSAT
X = df[['lsat','tier']]
km_x = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km_x.fit_predict(X)
index1 = 0
index2 = 1
# plot the w clusters
plt.scatter( X[y_km==0].iloc[:,index1], X[y_km==0].iloc[:,index2], s=50, c='blue', marker='s', edgecolor='black', label='cluster 1' )
plt.scatter( X[y_km==1].iloc[:,index1], X[y_km==1].iloc[:,index2], s=50, c='red', marker='o', edgecolor='black', label='cluster 2' )
# plot the centroids
plt.scatter(km_x.cluster_centers_[:, index1], km_x.cluster_centers_[:, index2], s=250, marker='*', c='yellow', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel("LSAT Score")
plt.ylabel("School Tier")
plt.grid()
plt.show()

# Plot for GPA & LSAT
X = df[['ugpa','gpa']]
km_x = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km_x.fit_predict(X)
index1 = 0
index2 = 1
# plot the w clusters
plt.scatter( X[y_km==0].iloc[:,index1], X[y_km==0].iloc[:,index2], s=50, c='blue', marker='s', edgecolor='black', label='cluster 1' )
plt.scatter( X[y_km==1].iloc[:,index1], X[y_km==1].iloc[:,index2], s=50, c='red', marker='o', edgecolor='black', label='cluster 2' )
# plot the centroids
plt.scatter(km_x.cluster_centers_[:, index1], km_x.cluster_centers_[:, index2], s=250, marker='*', c='yellow', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel("Undergraduate GPA")
plt.ylabel("Law School GPA")
plt.grid()
plt.show()

# %%

# %%
