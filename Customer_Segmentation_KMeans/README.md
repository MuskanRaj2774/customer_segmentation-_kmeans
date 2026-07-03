# PROJECT 9: Customer Segmentation using K-Means

**Domain**: Unsupervised Learning – Clustering
**Difficulty**: Beginner to Intermediate
**Author**: muskan raj

## Project Objective
Identify different customer groups using K-Means clustering and provide business insights for targeted marketing and personalized promotions.

## Dataset
Use the Kaggle dataset: `Mall_Customers.csv`
Features include CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).

## Project Structure
```text
Customer_Segmentation_KMeans/
│
├── README.md                  # Project overview and instructions
├── requirements.txt           # Python dependencies
├── customer_segmentation.ipynb# Interactive Jupyter Notebook
├── customer_segmentation.py   # Python execution script
├── dataset/                   # Contains original and clustered datasets
├── images/                    # Saved visualizations from EDA and Clustering
└── report.md                  # Business insights and conclusion
```

## Setup & Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python Script:
   ```bash
   python customer_segmentation.py
   ```
   This will output the logs to the console and save visualizations in the `images/` folder.

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook customer_segmentation.ipynb
   ```
   Follow along the cells for an interactive experience.

## Steps Covered
- **Exploratory Data Analysis (EDA)**: Understanding data distributions and relationships.
- **Data Cleaning & Preprocessing**: Handling categorical variables and standardizing features.
- **Feature Engineering**: Creating insightful metrics like `Loyalty_Score` and `Income_to_Spending_Ratio`.
- **K-Means Clustering**: Finding optimal K using the Elbow method and Silhouette score.
- **Cluster Evaluation**: Analyzing clusters using Inertia, Silhouette Score, and Davies-Bouldin Index.
- **Cluster Visualization**: 2D, 3D, and PCA scatter plots.
- **Business Recommendations**: Actionable insights for each customer segment.
