# Computational Tools for Data Science

## Dataset

This project uses the [Yelp Open Dataset](https://www.yelp.com/dataset). The data is **not** included in this repository: it is ~4 GB (above GitHub's file size limits) and Yelp's Terms of Use do not allow redistribution.

### Setup

1. Download the JSON version of the dataset from https://www.yelp.com/dataset (you must accept Yelp's Terms of Use).
2. Extract `yelp_dataset.tar` into a `data/` folder at the root of this repository:

   ```bash
   mkdir data
   tar -xf yelp_dataset.tar -C data
   ```

3. You should end up with:

   ```
   data/
   ├── yelp_academic_dataset_business.json
   ├── yelp_academic_dataset_checkin.json
   ├── yelp_academic_dataset_review.json
   ├── yelp_academic_dataset_tip.json
   └── yelp_academic_dataset_user.json
   ```

The `data/` folder is listed in `.gitignore`, so it will never be committed.

## Steps

1-Data: loading, cleaning, filtering, exploration; owns the notebook.

2-LSH: MinHash, banding, brute-force comparison.

3-Graph: Louvain, Leiden, spectral clustering, community analysis.

4-PCY: PCY and the A-priori comparison, rules for each community.

5-Evaluation: permutation tests, the influence test, final figures; also edits the report.

## Pipeline

1. **Data cleaning & preprocessing**
   - Filter to a single city.
   - Keep users with ≥10 reviews.
   - Keep the largest connected component of the friendship graph.
   - Normalize star ratings per user (subtract user mean).

2. **Friend communities**
   - Build the user–user friendship graph.
   - Detect communities with Louvain; evaluate with modularity.

3. **Taste communities**
   - Represent each user as the set of businesses (or categories) they reviewed.
   - Use MinHash + LSH to find user pairs with high Jaccard similarity.
   - Build a taste-similarity graph from those pairs and apply Louvain.
   - (Optional) Compare with k-means on per-category rating vectors.

4. **Comparison**
   - Measure overlap between friend and taste partitions with NMI and ARI.
   - Permutation test: check whether friends share tastes more than random user pairs.

5. **Evaluation & conclusions**
   - Report whether friend communities are also taste communities.
   - Compare LSH parameters and community detection methods (Louvain vs. spectral).
   - Discuss limitations (e.g., correlation vs. causation, data sparsity).

## Scope of the project
We are aiming to know if _friend communities_ are also _taste communities_
