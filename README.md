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

### 1. Data cleaning & preprocessing — Together

- Filter to a single city.
- Keep users with ≥10 reviews.
- Keep the largest connected component of the friendship graph.
- Normalize star ratings per user (subtract user mean).

### 2. Friend communities — Mattia

- Build the user–user friendship graph.
- Detect communities with Louvain; evaluate with modularity.

### 3. Taste communities — Cesi

- Represent each user as the set of businesses (or categories) they reviewed.
- Use MinHash + LSH to find user pairs with high Jaccard similarity.
- Build a taste-similarity graph from those pairs and apply Louvain.
- (Optional) Compare with k-means on per-category rating vectors.

### 4. Frequent itemsets & association rules (PCY vs. A-priori) — Mariona and Viktor

- Treat each user as a basket of the categories (or businesses) they rated positively.
- Mine frequent itemsets with both A-priori and PCY.
- Compare the two algorithms: runtime, memory use, and number of candidate pairs.
- Extract association rules for each friend community and each taste community.
- Compare rules across communities: do friend communities share the same rules as taste communities?

### 5. Comparison — Jose

- Measure overlap between friend and taste partitions with NMI and ARI.
- Permutation test: check whether friends share tastes more than random user pairs.

### 6. Evaluation & conclusions — Together

- Report whether friend communities are also taste communities.
- Compare LSH parameters and community detection methods (Louvain vs. spectral).
- Discuss limitations (e.g., correlation vs. causation, data sparsity).

## Scope of the project

We aim to investigate whether *friend communities* are also *taste communities*.

In other words, we want to understand whether users who are connected through Yelp's friendship network also tend to share similar preferences in businesses and categories.
