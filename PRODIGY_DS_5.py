import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('G:/Pratik Profile PC/Desktop/PRODIGY INFOTECH INTERNSHIP TASKS/TASK_05/US_Accidents_March23.csv')

df

#Creating User Columns
#df_user = pd.DataFrame(np.arange(0,len(df)), columns=['user'])
#df = pd.concat([df_user, df], axis=1)

df.head(5)

df.tail()

df.columns.values

df.isna().sum()

df.describe().T

df.describe()

df.info()

df['y'].value_counts()


# Find the number of columns that are numeric 
numerics = ['int16','int32','int64','float16','float32','float64']
numeric_df = df.select_dtypes(numerics)
len(numeric_df.columns)

# Find number of missing values in dataset 
missing_percentages = round(df.isnull().sum().sort_values(ascending=True) /len(df) *100,2)

missing_per = missing_percentages[missing_percentages.values > 0]
plt.figure(figsize=(10,7))
sns.barplot(x=missing_per , y= missing_per.index)
plt.xlabel('Missing Percentage')
plt.ylabel('Features')
plt.title('Missing Data Percentage by Feature')
plt.show()

# Analyzing the data by state column
states = df['State'].value_counts().head()
plt.figure(figsize=(10,7))
sns.barplot(y=states , x = states.index, palette="RdPu")
plt.title('Top 5 highest accident States')
plt.xlabel('State')
plt.ylabel('Count')
plt.show()

# Analyzing the cities columns 
cities_by_accidents = df['City'].value_counts()
df['City'].nunique()

city = cities_by_accidents.sort_values(ascending=False).head(20)
plt.figure(figsize=(10,7))
sns.barplot(x=city.values, y=city.index, color='green')
plt.xlabel('Number of Accidents')
plt.ylabel('City')
plt.title('Top 20 Cities with Highest Number of Accidents', y=1.05)
plt.show()

sns.set_style('whitegrid')
sns.distplot(cities_by_accidents) # Based on the diagram we see that the probability of accident occuring is very less 
plt.title("Number of accidents distributed across the cities")
plt.show()

# Analyzing the start time column 
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
# Accident over time 
plt.figure(figsize=(10,5))
sns.barplot(x = df['Start_Time'].dt.hour.value_counts().index,y = (df['Start_Time'].dt.hour.value_counts().values/len(df))*100, palette='pastel')
plt.title('Count of Accidents over time')
plt.show()

# Accident over day of week 
plt.figure(figsize=(10,5))
sns.barplot(x = df['Start_Time'].dt.day_of_week.value_counts().index,y = (df['Start_Time'].dt.day_of_week.value_counts().values/len(df))*100, palette='icefire')
plt.title('Count of Accidents over week days')
plt.show()
