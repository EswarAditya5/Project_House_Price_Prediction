import pandas as pd
import numpy as np
import streamlit as st

trainhp=pd.read_csv('train.csv')

trainhp.info() 

testhp=pd.read_csv('test.csv')

testhp.info()

testhp['SalePrice']='test'

combinedf=pd.concat([trainhp,testhp],axis=0)

combinedf.info()

combinedf.isnull().sum().sort_values(ascending=False)/combinedf.shape[0]*100

# creating a list which we need to impute "NOtAvailable" in missing value columns
notavailable=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
              'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']

# finding the length of the list
len(notavailable)

# Missing values imputation with the help of for loop 
for col in notavailable:
    combinedf[col]=combinedf[col].fillna('NotAvailable')

# Checking columns details like column names, not-null, values and data types
combinedf.info()

#split the data into numeric and object columns
numcols=combinedf.select_dtypes(include=np.number)
objcols=combinedf.select_dtypes(include=['object'])

#checking null values and null_values count
objcols.isnull().sum().sort_values(ascending=False)

# missing values imputation with idxmax() for objcols by using for loop
for col in objcols.columns:
    objcols[col]=objcols[col].fillna(objcols[col].value_counts().idxmax())
#idxmax() identifies the class or index of maximum frequency in value_counts().

# finding missing values
objcols.isnull().sum().sort_values(ascending=False)

#checking null values and null_values count
numcols.isnull().sum().sort_values(ascending=False)

objcols['OverallQual']=numcols['OverallQual'].astype('object')
objcols['OverallCond']=numcols['OverallCond'].astype('object')
objcols['YearBuilt']=numcols['YearBuilt'].astype('object')
objcols['YearRemodAdd']=numcols['YearRemodAdd'].astype('object')
objcols['GarageYrBlt']=numcols['GarageYrBlt'].astype('object')
objcols['MoSold']=numcols['MoSold'].astype('object')
objcols['YrSold']=numcols['YrSold'].astype('object')


# droping the columns which are added in objcols
numcols=numcols.drop(['OverallQual','OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold'],axis=1)

numcols.LotFrontage.value_counts(dropna=False)

#missing values impetation by using for loop
for col in numcols.columns:
    numcols[col]=numcols[col].fillna(numcols[col].median())


# Standard scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
numcols_scaled=sc.fit_transform(numcols)
numcols_scaled=pd.DataFrame(numcols_scaled,columns=numcols.columns)

# MinmAx Scaler
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
numcols_mm=mm.fit_transform(numcols)
numcols_mm=pd.DataFrame(numcols_mm,columns=numcols.columns)

# Robust Scaler
from sklearn.preprocessing import RobustScaler
robust=RobustScaler()
numcols_robust=robust.fit_transform(numcols)
numcols_robust=pd.DataFrame(numcols_robust,columns=numcols.columns)

objcols.info()

objcols=objcols.drop('SalePrice',axis=1)

# Label encoding for objcols
from sklearn.preprocessing import LabelEncoder
object_le=objcols.apply(LabelEncoder().fit_transform)
object_le.info()


# column wise concatination of numcols_mm,object_le,salesPrices columns
combinedf_clean=pd.concat([numcols_mm.reset_index(),object_le.reset_index(),combinedf.SalePrice.reset_index()],axis=1)

combinedf_clean.info()

# Dropping un wanted columns for predicting
combinedf_clean=combinedf_clean.drop(['index','Id'],axis=1)

# split data back to train and test
housetrain_df=combinedf_clean[combinedf_clean.SalePrice!='test']
housetest_df=combinedf_clean[combinedf_clean.SalePrice=='test']

# drop the dependent variable from test data
housetest_df=housetest_df.drop('SalePrice',axis=1)

# split data into dependent variable(y) and independent variable(X)
y=housetrain_df.SalePrice
X=housetrain_df.drop('SalePrice',axis=1)

# converting datatype of dependent variable(y) from object to int64
y=y.astype('int64')

# Model Building
# Linear Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
regmodel=reg.fit(X,np.log(y))
regmodel.score(X,np.log(y)) # r_square
regpred_X=regmodel.predict(X)
regresid=y-regpred_X # residual
np.sqrt(np.mean(regresid**2)) # RMSE
regpredict=regmodel.predict(housetest_df)

# Gradient Boosting Machine
from sklearn.ensemble import GradientBoostingRegressor
gbm=GradientBoostingRegressor(max_depth=1)
gbmmodel=gbm.fit(X,np.log(y))
gbmmodel.score(X,np.log(y)) #r_square
gbmpredict_X=gbmmodel.predict(X)
gbmresid=y-gbmpredict_X #residual
np.sqrt(np.mean(gbmpredict_X**2)) #RMSE
gbmpredict=gbmmodel.predict(housetest_df)

# ------------------------------------------------------------------------------------------------------


# App building

st.title('House Price Prediction')

# st.markdown('Model Predict Charges')

# House Details
st.header('House Details')
col1,col2,col3=st.columns(5)
with col1:
  age = st.slider('age',2,100,1)
  MSSubClass=st.input('MSSubClass')
with col2:
  bmi = st.slider('bmi',1,60,4)
with col3:
  children = st.slider('children',0,5,0)

 
 SalePrice  the property's sale price in dollarss
MSZoning The general zoning classification
LotFrontage Linear feet of street connected to property
LotArea Lot size in square feet
Street Type of road access
Alley Type of alley access
LotShape General shape of property
LandContour Flatness of the property
Utilities Type of utilities available
LotConfig Lot configuration
LandSlope Slope of property
Neighborhood Physical locations within Ames city limits
Condition1 Proximity to main road or railroad
Condition2 Proximity to main road or railroad (if a second is present)
BldgType Type of dwelling
HouseStyle Style of dwelling
OverallQual Overall material and finish quality
OverallCond Overall condition rating
YearBuilt Original construction date
YearRemodAdd Remodel date
RoofStyle Type of roof
RoofMatl Roof material
Exterior1st Exterior covering on house
Exterior2nd Exterior covering on house (if more than one material)
MasVnrType Masonry veneer type
MasVnrArea Masonry veneer area in square feet
ExterQual Exterior material quality
ExterCond Present condition of the material on the exterior
Foundation Type of foundation
BsmtQual Height of the basement
BsmtCond General condition of the basement
BsmtExposure Walkout or garden level basement walls
BsmtFinType1 Quality of basement finished area
BsmtFinSF1 Type 1 finished square feet
BsmtFinType2 Quality of second finished area (if present)
BsmtFinSF2 Type 2 finished square feet
BsmtUnfSF Unfinished square feet of basement area
TotalBsmtSF Total square feet of basement area
Heating Type of heating
HeatingQC Heating quality and condition
CentralAir Central air conditioning
Electrical Electrical system
1stFlrSF First Floor square feet
2ndFlrSF Second floor square feet
LowQualFinSF Low quality finished square feet (all floors)
GrLivArea Above grade (ground) living area square feet
BsmtFullBath Basement full bathrooms
BsmtHalfBath Basement half bathrooms
FullBath Full bathrooms above grade
HalfBath Half baths above grade
Bedroom Number of bedrooms above basement level
Kitchen Number of kitchens
KitchenQual Kitchen quality
TotRmsAbvGrd Total rooms above grade (does not include bathrooms)
Functional Home functionality rating
Fireplaces Number of fireplaces
FireplaceQu Fireplace quality
GarageType Garage location
GarageYrBlt Year garage was built
GarageFinish Interior finish of the garage
GarageCars Size of garage in car capacity
GarageArea Size of garage in square feet
GarageQual Garage quality
GarageCond Garage condition
PavedDrive Paved driveway
WoodDeckSF Wood deck area in square feet
OpenPorchSF Open porch area in square feet
EnclosedPorch Enclosed porch area in square feet
3SsnPorch Three season porch area in square feet
ScreenPorch Screen porch area in square feet
PoolArea Pool area in square feet
PoolQC Pool quality
Fence Fence quality
MiscFeature Miscellaneous feature not covered in other categories
MiscVal $Value of miscellaneous feature
MoSold Month Sold
YrSold Year Sold
SaleType Type of sale
SaleCondition Condition of sale






























































