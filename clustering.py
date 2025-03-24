from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_optimal_cluster(data):
    wcss=[]
    for i in range(1, 11):
        kmeans= KMeans(n_clusters=i, init= "k-means++", random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.show()

#Applying k-means clustering
def apply_kmeans(data, n_clusters=5):
    kmeans= KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    clusters= kmeans.fit_predict(data)
    return clusters

#Visualizing the clusters
def plot_clusters(data, clusters):
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Customer Segments")
    plt.colorbar(label="Cluster")
    plt.show()