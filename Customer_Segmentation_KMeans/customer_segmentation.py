import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------
# Setup and Configuration
# ------------------------------------------------
# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset/Mall_Customers.csv")
IMG_DIR = os.path.join(BASE_DIR, "images/")
os.makedirs(IMG_DIR, exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Starting Customer Segmentation Project...")

# ------------------------------------------------
# STEP 1: Exploratory Data Analysis
# ------------------------------------------------
print("\n--- STEP 1: Exploratory Data Analysis ---")
df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape}")
print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print(f"\nDuplicates: {df.duplicated().sum()}")

print("\nStatistical Summary:")
print(df.describe())

# 1. Histograms / Distribution Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Age Distribution')
sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, ax=axes[1], color='lightgreen')
axes[1].set_title('Annual Income Distribution')
sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=axes[2], color='salmon')
axes[2].set_title('Spending Score Distribution')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}histograms_distribution.png")
plt.show()

# 2. Count plot for Gender
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x='Gender', palette='pastel')
plt.title('Gender Count')
plt.savefig(f"{IMG_DIR}gender_countplot.png")
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig(f"{IMG_DIR}correlation_heatmap.png")
plt.show()

# 4. Pairplot
sns.pairplot(df.drop('CustomerID', axis=1), hue='Gender', palette='Set2')
plt.savefig(f"{IMG_DIR}pairplot.png")
plt.show()

# 5. Scatter Plot (Income vs Spending)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', palette='Set1', s=100)
plt.title('Income vs Spending Score')
plt.savefig(f"{IMG_DIR}scatter_income_spending.png")
plt.show()

# ------------------------------------------------
# STEP 2: Data Cleaning & Preprocessing
# ------------------------------------------------
print("\n--- STEP 2: Data Cleaning & Preprocessing ---")
# Drop CustomerID as it's not useful for clustering
data = df.drop(['CustomerID'], axis=1).copy()

# Encode Gender
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
print("Gender mapped (0=Female, 1=Male or vice versa).")

# ------------------------------------------------
# STEP 3: Feature Engineering
# ------------------------------------------------
print("\n--- STEP 3: Feature Engineering ---")
# Creating new features
# 1. Spending_per_Age: Represents spending behavior relative to age.
data['Spending_per_Age'] = data['Spending Score (1-100)'] / data['Age']

# 2. Income_to_Spending_Ratio: How much they spend relative to income.
data['Income_to_Spending_Ratio'] = data['Annual Income (k$)'] / (data['Spending Score (1-100)'] + 1e-5)

# 3. Loyalty_Score: Derived metric assuming higher age and higher spending relates to loyalty.
data['Loyalty_Score'] = (data['Age'] * 0.2) + (data['Spending Score (1-100)'] * 0.8)

print(data.head())

# Select features for clustering - using core features as requested
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data normalized successfully.")

# ------------------------------------------------
# STEP 4: K-Means Clustering
# ------------------------------------------------
print("\n--- STEP 4: K-Means Clustering ---")
wcss = []
silhouette_scores = []
K_RANGE = range(2, 11)

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method (WCSS)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_RANGE, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method (WCSS)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(K_RANGE, silhouette_scores, marker='s', linestyle='-', color='g')
plt.title('Silhouette Score Analysis')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.savefig(f"{IMG_DIR}elbow_silhouette_plots.png")
plt.show()

# Optimal K (Usually 5 for Mall Customers)
optimal_k = 5
print(f"Optimal K selected: {optimal_k}")

# Train K-Means
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)
df['Cluster'] = cluster_labels
data['Cluster'] = cluster_labels

# ------------------------------------------------
# STEP 5: Cluster Evaluation
# ------------------------------------------------
print("\n--- STEP 5: Cluster Evaluation ---")
inertia = kmeans_final.inertia_
sil_score = silhouette_score(X_scaled, cluster_labels)
db_index = davies_bouldin_score(X_scaled, cluster_labels)

print(f"Inertia (WCSS): {inertia:.2f}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")

# ------------------------------------------------
# STEP 6: Cluster Visualization
# ------------------------------------------------
print("\n--- STEP 6: Cluster Visualization ---")
# 1. 2D Scatter Plot (Income vs Spending)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100)
# Plot centroids
centroids_2d = scaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(centroids_2d[:, 1], centroids_2d[:, 2], s=300, c='red', marker='X', label='Centroids')
plt.title('Clusters: Income vs Spending Score')
plt.legend()
plt.savefig(f"{IMG_DIR}cluster_2d_scatter.png")
plt.show()

# 2. 3D Scatter Plot
fig = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
                    color='Cluster', opacity=0.8, title='3D Cluster Visualization',
                    color_continuous_scale=px.colors.sequential.Viridis)
fig.write_image(f"{IMG_DIR}cluster_3d_scatter.png")
fig.show()

# 3. PCA-based visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='tab10', s=100)
plt.title('PCA - 2D Projection of Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig(f"{IMG_DIR}pca_clusters.png")
plt.show()

# 4. Cluster Distribution Chart
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x='Cluster', palette='Set3')
plt.title('Number of Customers per Cluster')
plt.savefig(f"{IMG_DIR}cluster_distribution.png")
plt.show()

# ------------------------------------------------
# STEP 7: Cluster Analysis
# ------------------------------------------------
print("\n--- STEP 7: Cluster Analysis ---")
cluster_analysis = df.groupby('Cluster').agg({
    'Age': 'mean',
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'Size'}).reset_index()

print(cluster_analysis.round(2))

# Save processed dataset
df.to_csv(os.path.join(BASE_DIR, "dataset/Mall_Customers_Clustered.csv"), index=False)
print("\nProject executed successfully. Outputs saved in dataset/ and images/ directories.")
