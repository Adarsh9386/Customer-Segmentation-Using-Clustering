# Loading the dataset
import pandas as pd

#Data Preprocessing
def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully....!")
    print(data.head())
    return data

#Data Cleaning
def clean_data(df):
    print("Missing values: \n", df.isnull().sum())
    df= df.drop_duplicates()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print("Data cleaned successfully!")
    return df

#Data Scaling
from sklearn.preprocessing import StandardScaler
def scale_data(df, features):
    scaler= StandardScaler()
    df_scaled= scaler.fit_transform(df[features])
    print("Data scaling complete!")
    print("Scaled data (First 10 rows): \n", df_scaled[:10])
    return df_scaled

#File path and execution
file_path= "C:/Users/AESTHETIC/Desktop/Mall_Customers.csv"
data= load_data(file_path)
cleaned_data= clean_data(data)

#Defining features to scale
features= ["Annual Income (k$)", "Spending Score (1-100)"]
scaled_data= scale_data(cleaned_data, features)