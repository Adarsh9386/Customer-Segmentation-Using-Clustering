from preprocessing import load_data, clean_data, scale_data
from visualization import plot_data
from clustering import find_optimal_clusters, apply_kmeans, plot_clusters
import seaborn as sns

# Load and clean data
file_path = "C:/Users/AESTHETIC/Desktop/Mall_Customers.csv"
df = load_data(file_path)
df = clean_data(df)

# Select features and scale data
features = ['Annual Income (k$)', 'Spending Score (1-100)']
scaled_data = scale_data(df, features)

# Perform EDA
plot_data(df)

# Find optimal clusters
find_optimal_clusters(scaled_data)

# Apply KMeans and visualize
clusters = apply_kmeans(scaled_data, n_clusters=5)
plot_clusters(scaled_data, clusters)


ab=sns.pairplot(df, hue=True)
print(ab)