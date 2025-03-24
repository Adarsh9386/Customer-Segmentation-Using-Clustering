import sys
sys.path.append("C:/Users/AESTHETIC/Desktop/CST/Customer_Segmentation/preprocessing.py")
from preprocessing import cleaned_data
# print(cleaned_data.head())

#EDA - exploratory data analysis
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data(df):
    sns.pairplot(df)
    plt.show()
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()
plot_data(cleaned_data)