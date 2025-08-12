# Clustering — Grouping Unlabelled Data

<figure style="text-align: center;">
  <img src="https://www.mdpi.com/symmetry/symmetry-10-00734/article_deploy/html/images/symmetry-10-00734-g001-550.jpg" alt="Supervised and Unsupervised Machine Learning" width="500">
  <figcaption><strong>Fig 1.</strong> Supervised and Unsupervised Machine Learning</figcaption>
</figure>

---

## The Elbow Method for Optimal K

The **Elbow Method** is a technique used to determine the optimal number of clusters (**K**) in K-means clustering.

### Theory
- In K-means, the objective is to minimize the **Within-Cluster Sum of Squares (WCSS)** — the total squared distance between each point and the centroid of its cluster.
- As **K** increases:
  - WCSS decreases because clusters become smaller and more specific.
  - However, after a certain **K**, the marginal improvement in WCSS becomes minimal.
- When plotting **WCSS vs K**, the curve usually forms a bend resembling an **elbow**.
- The point where the decrease in WCSS slows significantly is considered the **optimal K**.

**Mathematical Formulation:**
\[
WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} ||x - \mu_i||^2
\]
Where:
- \( C_i \) = set of points in cluster \( i \)  
- \( \mu_i \) = centroid of cluster \( i \)  

<figure style="text-align: center;">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif" alt="K-means Clustering Example" width="400">
  <figcaption><strong>Fig 2.</strong> K-means Clustering Example</figcaption>
</figure>

---

## Issue with Random Initialization in K-means

In **standard K-means**, the initial centroids are chosen randomly from the dataset.

### Problems:
- **Poor convergence**: Initial centroids too close together may lead to overlapping or suboptimal clusters.
- **Inconsistent results**: Different runs may yield different final clusters.
- **Slow convergence**: Bad starting points can require more iterations to stabilize.

---

## K-means++ Initialization

**K-means++** is an improved centroid initialization technique that reduces the likelihood of poor clustering outcomes.

### Algorithm Steps:
1. Randomly choose the first centroid from the data points.
2. For each remaining data point, compute the **distance squared** from the nearest chosen centroid.
3. Choose the next centroid with a probability proportional to this squared distance (favoring points far from existing centroids).
4. Repeat until **K** centroids are chosen.
5. Proceed with standard K-means iterations.

### Advantages:
- Ensures **spread out** initial centroids.
- Improves clustering accuracy.
- Often converges **faster**.
- Produces more consistent results across runs.
