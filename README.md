# Fashion Dataset Clustering Analysis

## Overview
This project involves clustering fashion data from Paris and New York Fashion Weeks using hierarchical clustering and analyzing the trend patterns over the years. The dataset contains categorical features such as Colour, Patterns, Shoes, Accessories, Fabric, and Theme, which are one-hot encoded and weighted before performing clustering.

## Dataset
- **Source:** `Fashion Dataset - Sheet1.csv`
- **Filtered Locations:** Paris, New York
- **Excluded Columns:** Year, Collection, Location, Gender, Collection Type
- **Categorical Features:** Colour, Patterns, Shoes, Accessories, Fabric, Theme

## Preprocessing
- Data is filtered for Paris and New York Fashion Weeks.
- Categorical features are one-hot encoded using `MultiLabelBinarizer`.
- Feature weighting is applied (weight = 80 for each feature).

## Clustering Methodology
- Hierarchical clustering with `ward` linkage method is performed.
- The dataset is clustered into **5 clusters**.
- Predicted clusters are added to the dataset.

## Visualizations
1. **Dendrogram**
   - Displays the hierarchical clustering results.
   - Cut-off applied to form 5 clusters.
   - Saved as `dendogram2.png`.

2. **Cluster Trends Over Years**
   - Line plots are generated for each `Collection Type` (Ready-to-Wear, Menswear, Couture).
   - Crossovers between Paris and New York are identified and marked.
   - Correlation between clusters across locations is calculated.
   - Saved as `<collection_type>_clusters_over_years.png`.

3. **Cluster Distribution**
   - Bar chart representing the number of data points in each cluster.
   - Labels include cluster descriptions:
     - **Cluster 1:** Funky, chunky jewelry, denim
     - **Cluster 2:** Rich, houndstooth, boots, woolen
     - **Cluster 3:** Pop, abstract, flats, organza
     - **Cluster 4:** Academia, stripes, loafers, linen
     - **Cluster 5:** Classy, plain, pointed pumps, silk
   - Displays count per cluster.

4. **Closest Points to Cluster Centroids**
   - Identifies the closest data point to each cluster centroid.
   - Displays characteristics of the closest data point for each cluster.

5. **Jaccard Distance Matrix**
   - Computes pairwise Jaccard distances between cluster centroids.
   - Visualized using a heatmap (`jaccard_distance_matrix.png`).

## Outputs
- **Clustering Results:** Predicted clusters stored in the dataset.
- **Visualizations:**
  - Dendrogram (`dendogram2.png`)
  - Yearly cluster trends per collection type (`<collection_type>_clusters_over_years.png`)
  - Cluster distribution plot
  - Jaccard distance heatmap (`jaccard_distance_matrix.png`)

## Dependencies
- `pandas`
- `numpy`
- `sklearn.preprocessing`
- `scipy.cluster.hierarchy`
- `scipy.spatial.distance`
- `matplotlib`
- `seaborn`

## Running the Code
1. Ensure the dataset (`Fashion Dataset - Sheet1.csv`) is available in the working directory.
2. Run the script to perform clustering and generate visualizations.
3. Review the generated plots and analyze clustering results.

## Future Improvements
- Experiment with different clustering methods (e.g., K-Means, DBSCAN).
- Implement automated feature selection for weighting.
- Expand the dataset to include more locations and years.

---
**Author:** Shivani Parab  
**Date:** November 2024

