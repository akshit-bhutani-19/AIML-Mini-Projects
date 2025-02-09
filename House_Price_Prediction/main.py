import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Read the dataset from a CSV file and load it into a Pandas DataFrame (df)
df = pd.read_csv('Housing.csv')


#Explore Database
print(df.head())        #Show first five rows
print(df.describe())    #Get summary statistics (mean, standard deviation, min, max, quartiles) for numerical columns
print(df.isna().sum())  #Checking for missing values in each column, helping identify if data cleaning is needed.



#Data Preprocessing

columns_to_transform = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[columns_to_transform] = df[columns_to_transform].replace({'yes': 1, 'no': 0})            #Binary Encoding yes/no to 1/0

df['furnishingstatus'] = df['furnishingstatus'].replace({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})   #Ordinal Encoding


scaler = StandardScaler()                                #Normalize price and area to mean of 0 and standard deviation of 1
sc = ['price', 'area']                                   
df[sc] = scaler.fit_transform(df[sc])


#Check dataset after preprocessing

print(df)
print(df.dtypes)


#Check for correlation between different variables to check for relation between them
# corr_matrix = df.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
# plt.show()

#Vizualize data through histogram
# df.hist(figsize=(10, 10), bins=10)
# plt.suptitle("Histograms for All Columns", fontsize=16) 
# plt.show()



#Visualize area, bedroom and bathroom wrt price to check if they have linear relation 
X = df.drop('price', axis=1)
y = df['price']

X_features = ['area', 'bedrooms','bathrooms']

fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(3):  # Assuming there are 4 features
    ax[i].scatter(X.iloc[:, i], y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price z-score Normalized")
plt.show()