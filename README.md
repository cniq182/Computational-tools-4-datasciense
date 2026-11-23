# Computational Tools for Data Science

## Dataset

This project uses the [Yelp Open Dataset](https://www.yelp.com/dataset). The original dataset contains several JSON files. Only the three datasets needed for this project are used:

- `business.json`
- `review.json`
- `user.json`

The dataset is **not included in this repository**. Yelp's dataset is approximately 4 GB, and its Terms of Use do not allow redistribution.

### Setup

1. Download the JSON version of the dataset from https://www.yelp.com/dataset. You must accept Yelp's Terms of Use.

2. Extract the downloaded `yelp_dataset.tar` archive:

   ```bash
   tar -xf yelp_dataset.tar
   ```
  The three files correspond to the following original Yelp dataset files:

   | Project filename | Original Yelp filename |
   |---|---|
   | `business.json` | `yelp_academic_dataset_business.json` |
   | `review.json` | `yelp_academic_dataset_review.json` |
   | `user.json` | `yelp_academic_dataset_user.json` |
   
3. Create the following folder structure in the root of the repository. Move the three required JSON files into `data/raw/`:

   ```text
   project/
   ├── data/
   │   └── raw/
   │       ├── business.json
   │       ├── review.json
   │       └── user.json
   │
   └── notebooks/
       └── 01_data_cleaning.ipynb
   ```

 

4. Install the required Python dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

5. Start Jupyter Notebook from the project root:

   ```bash
   jupyter notebook
   ```

6. Open and run `notebooks/01_data_cleaning.ipynb`, executing the cells in order.

**Important:** The notebook should reference the raw data files using paths relative to its location, for example:

```python
business_path = "../data/raw/business.json"
review_path = "../data/raw/review.json"
user_path = "../data/raw/user.json"
```

The `data/` folder is listed in `.gitignore` and should never be committed to the repository.

## Steps

1. **Data:** Loading, cleaning, filtering, and exploration; owns the notebook.
2. **LSH:** MinHash, banding, and brute-force comparison.
3. **Graph:** Louvain, Leiden, spectral clustering, and community analysis.
4. **PCY:** PCY and A-priori comparison, with rules for each community.
5. **Evaluation:** Permutation tests, the influence test, and final figures; also edits the report.

## Pipeline

### 1. Data cleaning & preprocessing — Together (`notebooks/01_data_cleaning.ipynb`)

- Filter to a single city: **Philadelphia, PA**, restaurants only (category "Restaurants").
- Keep one review per (user, restaurant) (the most recent).
- Keep users with ≥ 10 restaurant reviews.
- Keep the largest connected component of the friendship graph → **7,947 users, 251,375 reviews, 5,759 restaurants, 67,622 friendships**.
- Normalize star ratings per user (`stars_norm` = stars − user mean) and define "liked" as `is_positive` = stars ≥ 3.5 (4 or 5 stars).
- Outputs: `data/processed/philadelphia_pa_*.parquet` and `cleaning_summary.json`.

### 2. Friend communities — Mattia (`notebooks/02_friend_communities.ipynb`)

- Build the user–user friendship graph.
- Detect communities with **Louvain implemented from scratch** (20 seeds, best-modularity run, fix for internally disconnected communities); evaluate with modularity against degree-preserving random graphs.
- Analyse the communities, including their **period of activity** (a confounder for step 5).
- Outputs: `data/processed/philadelphia_pa_friend_communities*.parquet`. Details: `STEP2_FRIEND_COMMUNITIES.md`.

### 3. Taste communities — Cesia (`notebooks/03_taste_communities.ipynb`, `src/`)

- Represent each user as the set of **restaurants** they liked and, separately, the set of **categories** of those restaurants (without generic categories, rare categories and multi-category venues).
- Use **MinHash + LSH (implemented from scratch)** to find each user's 10 most similar users (Jaccard); compare with exact brute force (recall, runtime, choice of bands × rows).
- Build a weighted taste-similarity graph (k-nearest neighbours, weight = Jaccard) and apply the same Louvain as step 2 (extended to edge weights).
- Outputs: `data/results/taste_communities*.parquet`, `taste_edges_*.parquet`, `taste_summary.json`.
- (Optional, not done) Compare with k-means on per-category rating vectors.

### 4. Frequent itemsets & association rules (PCY vs. A-priori) — Mariona and Viktor

- Treat each user as a basket of the categories (or restaurants) they rated positively (`is_positive`, same definition as steps 1 and 3).
- Mine frequent itemsets with both A-priori and PCY.
- Compare the two algorithms: runtime, memory use, and number of candidate pairs.
- Extract association rules for each friend community and each taste community.
- Compare rules across communities: do friend communities share the same rules as taste communities?

### 5. Comparison — Jose

- Measure overlap between friend and taste partitions with NMI and ARI, for the chosen runs **and** across the 20 saved runs of each.
- Permutation test: check whether friends share tastes more than random user pairs, **controlling for the period of activity** (shuffle within the same period; `period` is in `taste_communities.parquet`).
- Compare the restaurant-based and category-based taste partitions (restaurant overlap between friends may reflect going together or being active in the same years).

### 6. Evaluation & conclusions — Together

- Report whether friend communities are also taste communities.
- Compare LSH parameters and community detection methods (Louvain vs. spectral).
- Discuss limitations (e.g., correlation vs. causation, data sparsity, selection of active connected users, moderate stability of Louvain partitions).