# Clustering - grouping unlabelled data

<figure style="text-align: center;">
  <img src="https://www.mdpi.com/symmetry/symmetry-10-00734/article_deploy/html/images/symmetry-10-00734-g001-550.jpg" alt="Supervised and Unsupervised Machine Learning">
  <figcaption>Fig 1. Supervised and Unsupervised Machine Learning</figcaption>
</figure>

## The Elbow Method for Optimal K

The **Elbow Method** is a technique used to determine the optimal number of clusters (**K**) in K-means clustering.

### Theory:
- In K-means, we aim to minimize the **Within-Cluster Sum of Squares (WCSS)** — the total squared distance between each point and the centroid of its cluster.
- As K increases:
  - WCSS decreases because clusters are smaller and more specific.
  - However, after a certain K, the marginal improvement in WCSS becomes very small.
- If we plot **WCSS vs K**, the curve will typically show a sharp bend (like an elbow).
- The point at which the WCSS reduction slows down significantly is called the **“elbow point”**, and this K is considered optimal.

Mathematically:
\[
WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2
\]

Where:
- \( C_i \) = set of points in cluster \( i \)  
- \( \mu_i \) = centroid of cluster \( i \)  

<figure style="text-align: center;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif" alt="K-means Clustering Example" width="400">
  <figcaption>Fig 2. K-means Clustering Example</figcaption>
</figure>
