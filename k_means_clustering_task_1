import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Read data from CSV file
df = pd.read_csv('customer_purchase_behavior.csv')

# Features to be used for clustering
features = [
    "TotalPurchaseAmount", "PurchaseFrequency", "AveragePurchaseValue",
    "DaysSinceLastPurchase", "ProductCategories", "TotalItemsPurchased",
    "AverageItemsPerPurchase", "ReturnsCount"
]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Perform K-Means clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualization of the clusters
plt.figure(figsize=(12, 7))
scatter_plot = sns.scatterplot(
    data=df, x='TotalPurchaseAmount', y='PurchaseFrequency', 
    hue='Cluster', palette='viridis'
)
scatter_plot.set_title('Customer Segmentation by Purchase Behavior')
scatter_plot.set_xlabel('Total Purchase Amount')
scatter_plot.set_ylabel('Purchase Frequency')
plt.legend(title='Cluster')
plt.show()
