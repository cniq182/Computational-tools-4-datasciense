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
