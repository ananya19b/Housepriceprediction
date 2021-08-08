#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns


import os


# In[22]:


train = pd.read_csv(r'C:\Users\anany\Desktop\Kaggle data/train.csv')
test = pd.read_csv(r'C:\Users\anany\Desktop\Kaggle data/test.csv')


# In[3]:


train.head()


# In[4]:


print(train.shape)


# In[5]:


num_cols = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
# Id & SalePrice 
num_cols.remove('Id')
num_cols.remove('SalePrice')
# Num_cols 
num_analysis = train[num_cols].copy()

for col in num_cols:
    if num_analysis[col].isnull().sum() > 0:
        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1,1))
        
# Model
# ExtraTressRegressor
clf = ExtraTreesRegressor(random_state=42)
reg_model = clf.fit(num_analysis, train.SalePrice)


# In[7]:


def plot_importance(model, features, num=len(num_cols), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features})
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(reg_model, num_cols)


# In[8]:


plt.figure(figsize=(8,8))
plt.title('Correlation Matrix')
cols =['OverallQual', 'GarageCars', 'GrLivArea', 'YearBuilt', 
       'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea','Fireplaces','GarageYrBlt','SalePrice']
sns.heatmap(train[cols].corr(),annot=True,square=True);


# In[9]:


def plot_numerical(col, discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' analysis')
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' analysis')


# In[10]:


plot_numerical('OverallQual',discrete=True);


# In[11]:


plot_numerical('GarageCars',discrete=True);


# In[12]:


plot_numerical('GrLivArea')


# In[13]:


cat_features = [col for col in train.columns if train[col].dtype =='object']

cat_analysis = train[cat_features].copy()

for col in cat_analysis:
    if cat_analysis[col].isnull().sum() > 0:
        cat_analysis[col] = SimpleImputer(strategy='constant').fit_transform(cat_analysis[col].values.reshape(-1,1))


# In[14]:


# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
cat_analysis = one_hot_encoder(cat_analysis,cat_features)


# In[15]:


# Model 
clf = ExtraTreesRegressor(random_state=42)
h = clf.fit(cat_analysis, train.SalePrice)


# In[16]:


def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:20])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(h, cat_analysis)


# In[17]:


cat_analysis["SalePrice"] = train["SalePrice"]


# In[19]:


def cat_plot(col1,col2):
    
    fig, ax = plt.subplots(1,2,figsize=(12,6), sharey=True)
    sns.stripplot(x=col1, y='SalePrice', data=train, ax=ax[0])
    sns.boxplot(x=col1, y='SalePrice', data=train, ax=ax[1])
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
    fig.suptitle(str(col1) + ' analysis')
    
    # one-hot encoding
    fig, ax = plt.subplots(1,2,figsize=(12,6), sharey=True)
    sns.stripplot(x=col2, y='SalePrice', data=cat_analysis, ax=ax[0])
    sns.boxplot(x=col2, y='SalePrice', data=cat_analysis, ax=ax[1])
    fig.suptitle(str(col2) + ' analysis')


# In[20]:


cat_plot("ExterQual","ExterQual_TA")


# In[23]:


df = pd.concat([train, test]).reset_index(drop=True)
print(df.shape)


# In[24]:


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
    
missing_values_table(df)


# In[25]:


none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']


# In[26]:


zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
             'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']


# In[27]:


freq_cols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
             'SaleType', 'Utilities']


# In[28]:


for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    df[col].replace(np.nan, 'None', inplace=True)
    
for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)


# In[29]:


missing_values_table(df)


# In[30]:


df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(
    lambda x: x.fillna(x.mode()[0]))

df['LotFrontage'] = df.groupby(
    ['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

missing_values_table(df)


# In[31]:


df['MSSubClass'] = df['MSSubClass'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df.info()


# In[32]:


#FeatureEngineering
df["LotShape"].value_counts()


# In[33]:


df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR1"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR1"
df["LotShape"].value_counts()


# In[34]:


df["ExterQual"].value_counts()


# In[35]:


df.loc[df["ExterQual"]=="Ex","ExterQual"]=2
df.loc[df["ExterQual"]=="Gd","ExterQual"]=2
df.loc[df["ExterQual"]=="TA","ExterQual"]=1
df.loc[df["ExterQual"]=="Fa","ExterQual"]=1
df["ExterQual"]= df["ExterQual"].astype("int")

df["ExterQual"].value_counts()


# In[36]:


df["BsmtQual"].value_counts()


# In[37]:


df.loc[df["BsmtQual"]=="Ex","BsmtQual"]=2
df.loc[df["BsmtQual"]=="Gd","BsmtQual"]=2
df.loc[df["BsmtQual"]=="TA","BsmtQual"]=1
df.loc[df["BsmtQual"]=="Fa","BsmtQual"]=1
df.loc[df["BsmtQual"]=="None","BsmtQual"]=0
df["BsmtQual"]= df["BsmtQual"].astype("int")
df["BsmtQual"].value_counts()


# In[38]:


df.groupby("Neighborhood").agg({"SalePrice":"mean"}).sort_values(by="SalePrice", ascending=False)


# In[39]:


neigh_map = {'MeadowV': 1,'IDOTRR': 1,'BrDale': 1,'BrkSide': 2,'OldTown': 2,'Edwards': 2,
             'Sawyer': 3,'Blueste': 3,'SWISU': 3,'NPkVill': 3,'NAmes': 3,'Mitchel': 4,
             'SawyerW': 5,'NWAmes': 5,'Gilbert': 5,'Blmngtn': 5,'CollgCr': 5,
             'ClearCr': 6,'Crawfor': 6,'Veenker': 7,'Somerst': 7,'Timber': 8,
             'StoneBr': 9,'NridgHt': 10,'NoRidge': 10}

df['Neighborhood'] = df['Neighborhood'].map(neigh_map).astype('int')


# In[42]:


ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterCond'] = df['ExterCond'].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

df['BsmtCond'] = df['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {'None': 0,'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmf_map).astype('int')
df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['HeatingQC'] = df['HeatingQC'].map(heat_map).astype('int')

df['FireplaceQu'] = df['FireplaceQu'].map(bsm_map).astype('int')
df['GarageCond'] = df['GarageCond'].map(bsm_map).astype('int')
df['GarageQual'] = df['GarageQual'].map(bsm_map).astype('int')


# In[43]:


df["GarageCars"].value_counts()


# In[44]:


df["LotConfig"].value_counts()


# In[45]:


df.loc[(df["LotConfig"]=="Inside"),"LotConfig"] = 1
df.loc[(df["LotConfig"]=="FR2"),"LotConfig"] = 1
df.loc[(df["LotConfig"]=="Corner"),"LotConfig"] = 1

df.loc[(df["LotConfig"]=="FR3"),"LotConfig"] = 2
df.loc[(df["LotConfig"]=="CulDSac"),"LotConfig"] = 2
df["LotConfig"].value_counts()


# In[46]:


df["LandSlope"].value_counts()


# In[47]:


df["OverallQual"].value_counts()


# In[48]:


df.loc[df["OverallQual"] == 1, "OverallQual"] = 1
df.loc[df["OverallQual"] == 2, "OverallQual"] = 1
df.loc[df["OverallQual"] == 3, "OverallQual"] = 1
df.loc[df["OverallQual"] == 4, "OverallQual"] = 2
df.loc[df["OverallQual"] == 5, "OverallQual"] = 3
df.loc[df["OverallQual"] == 6, "OverallQual"] = 4
df.loc[df["OverallQual"] == 7, "OverallQual"] = 5
df.loc[df["OverallQual"] == 8, "OverallQual"] = 6
df.loc[df["OverallQual"] == 9, "OverallQual"] = 7
df.loc[df["OverallQual"] == 10, "OverallQual"] = 8
df["OverallQual"].value_counts()


# In[49]:


df["MasVnrType"].value_counts()


# In[50]:


df.loc[df["MasVnrType"] == "BrkCmn" , "MasVnrType"] = "None" 
df["MasVnrType"].value_counts()


# In[51]:


df["Foundation"].value_counts()


# In[52]:


df.loc[df["Foundation"] == "Stone", "Foundation"] = "BrkTil"
df.loc[df["Foundation"] == "Wood", "Foundation"] = "CBlock"
df["Foundation"].value_counts()


# In[53]:


df["Fence"].value_counts()


# In[54]:


df.loc[df["Fence"] == "MnWw", "Fence"] = "MnPrv"
df.loc[df["Fence"] == "GdWo", "Fence"] = "MnPrv"
df["Fence"].value_counts()


# In[56]:


#NewFeatures
df["TotalBath_NEW"] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5 + df['FullBath'] + df['HalfBath'] * 0.5


df['TotalSF_NEW'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])


df['TotalPorchSF_NEW'] = (df['OpenPorchSF'] + df['3SsnPorch'] +df['EnclosedPorch'] +df['ScreenPorch'] + df['WoodDeckSF'])


df["OVER_QUAL_NEW"] = df['OverallQual'] + df['OverallCond']


df["BSMT_QUAL_NEW"] = df['BsmtQual'] + df['BsmtCond']
df["EX_QUAL_NEW"] = df['ExterQual'] + df['ExterCond']


df['TotalGrgQual_NEW'] = (df['GarageQual'] + df['GarageCond'])


#df['TotalQual_NEW'] = df['OverallQual'] + df['EX_QUAL_NEW']  + df['TotalGrgQual_NEW'] + df['KitchenQual'] + df['HeatingQC']

df.loc[(df['Fireplaces'] > 0) & (df['GarageCars'] >= 3), "LUX_NEW"] = 1
df["LUX_NEW"].fillna(0, inplace=True)
df["LUX_NEW"] = df["LUX_NEW"].astype(int)
df.loc[df["YearBuilt"] == df["YearRemodAdd"], "NEW_home"] = 0
df.loc[df["YearBuilt"] != df["YearRemodAdd"], "NEW_home"] = 1

df['QualPorch_NEW'] = df['EX_QUAL_NEW'] * df['TotalPorchSF_NEW']

df['HasPool_NEW'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['Has2ndFloor_NEW'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['HasGarage_NEW'] = df['TotalGrgQual_NEW'].apply(lambda x: 1 if x > 0 else 0)
df['HasFireplace_NEW'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df['HasPorch_NEW'] = df['QualPorch_NEW'].apply(lambda x: 1 if x > 0 else 0)
df["Garden_NEW"]=df["LotArea"] - df["GrLivArea"]


# In[57]:


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# In[58]:


num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
for col in num_cols:
    replace_with_thresholds(df, col)


# In[59]:


for col in num_cols:
    print(col, check_outlier(df, col))


# In[60]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# In[61]:


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc,cat_cols):
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts()/len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe


# In[62]:


rare_analyser(df, "SalePrice", cat_cols)


# In[63]:


df = rare_encoder(df, 0.01, cat_cols)
rare_analyser(df, "SalePrice", cat_cols)


# In[64]:


useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.02).any(axis=None))]
useless_cols


# In[65]:


cat_cols = [col for col in cat_cols if col not in useless_cols]
df.shape


# In[66]:


for col in useless_cols:
    df.drop(col, axis=1, inplace=True)
df.shape


# In[67]:


rare_analyser(df, "SalePrice", cat_cols)


# In[68]:


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
df.shape


# In[69]:


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# In[70]:


rare_analyser(df, "SalePrice", cat_cols)


# In[71]:


useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]
useless_cols_new


# In[72]:


for col in useless_cols_new:
    df.drop(col, axis=1, inplace=True)
df.shape


# In[74]:


#FinalModel
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)


# In[84]:


#from catboost import CatBoostRegressor
#from lightgbm import LGBMRegressor
#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.svm import SVR
#from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor


# In[85]:


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet())]


# In[86]:


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# In[88]:


model = Ridge(alpha=1.0)
model.fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse


