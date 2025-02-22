# K-Means Clustering on Fashion-MNIST with and without PCA


This repository demonstrates the implementation of the **K-Means clustering algorithm** on the **Fashion-MNIST dataset**, both with and without dimensionality reduction using **Principal Component Analysis (PCA)**. The project is presented in a Jupyter Notebook, making it easy to follow and reproduce the experiments.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Implementation Details](#implementation-details)
   - [K-Means without PCA](#k-means-without-pca)
   - [K-Means with PCA](#k-means-with-pca)


---

## Overview

K-Means is an unsupervised learning algorithm used for clustering data into \( k \) groups. This project applies K-Means to the **Fashion-MNIST** dataset to group images of clothing into clusters without relying on labels. To evaluate the impact of dimensionality reduction, the algorithm is tested both with and without PCA.

**Key Objectives:**
1. Cluster images from the Fashion-MNIST dataset using K-Means.
2. Analyze the impact of dimensionality reduction using PCA on clustering performance.
3. Visualize and compare the results.

---

## Dataset

The **Fashion-MNIST dataset** consists of 70,000 grayscale images of clothing items, each with a resolution of \( 28 \times 28 \) pixels. The dataset is divided into 10 classes:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Each image is represented as a vector of 784 features (flattened from \( 28 \times 28 \)).

---

## Implementation Details

### K-Means Without PCA

- **Input Features**: Raw pixel values (784 dimensions).
- **Preprocessing**: 
  - Normalize pixel values to [0, 1].
  - Apply K-Means directly on the high-dimensional dataset.
- **Clustering**:
  - Experimented with different values of \( k \) (number of clusters).
  - Evaluated clusters using inertia and visual inspection.

### K-Means With PCA

- **Input Features**: Reduced dimensions after applying PCA.
- **Preprocessing**:
  - Normalize pixel values to [0, 1].
  - Apply PCA to reduce dimensionality while retaining 95% of the variance.
- **Clustering**:
  - Perform K-Means on the lower-dimensional dataset.
  - Compare results with the non-PCA version.

---

