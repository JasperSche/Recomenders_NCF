# Recomenders_NCF

## Dataset

`Dataset.py` contains the `NCFDataset` class which handles loading and preprocessing of the MovieLens `ratings.dat` file.

### Preprocessing steps

1. **Implicit conversion**
   - Ratings ≥ 4 → positive interactions
   - Ratings < 4 → ignored

2. **Train/validation/test split**
   - Positive interactions split per user (70/15/15)

3. **Negative sampling**
   - For each training positive, `num_neg` unseen items are sampled as negatives
   - Used only for training! (Evaluation on val/test is full-ranking, so all movies are used with positives as ground truth)

4. **Evaluation setup**
   - Validation/test store only positive items
   - Evaluation ranks all unseen items (full-ranking)