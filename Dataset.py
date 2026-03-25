import random
import torch
import statistics
from torch.utils.data import Dataset


class NCFDataset(Dataset):
   """
   1. reads ratings.dat
   2. keeps ratings >= threshold (=4) as positive interactions
   3. splits positives into train / val / test per user (70/15/15)
   4. negative-samples only for the training set

   Training samples returned by __getitem__:
      (user, item, label)

   Useful attributes:
      num_users
      num_items
      train_data
      val_data
      test_data
      train_matrix         # user -> set(train items)
      val_ground_truth     # user -> set(val items)
      test_ground_truth    # user -> set(test items)
   """

   def __init__(self, path="Data/ratings.dat", num_neg=4, threshold=4, seed=42):
      self.path = path
      self.num_neg = num_neg
      self.threshold = threshold
      self.seed = seed

      self.num_users = 0
      self.num_items = 0

      self.train_data = []
      self.val_data = []
      self.test_data = []

      self.train_matrix = {}
      self.val_ground_truth = {}
      self.test_ground_truth = {}

      self._load_data()

   def _load_data(self):
      random.seed(self.seed)

      # -------------------------------------------------
      # 1. Read positive interactions only
      # -------------------------------------------------
      user_pos = {}   # user -> set(items)
      max_user = 0
      max_item = 0

      with open(self.path, "r", encoding="latin-1") as f:
         for line in f:
               user, item, rating, _ = line.strip().split("::")
               user = int(user) - 1   # convert to 0-based
               item = int(item) - 1   # convert to 0-based
               rating = int(rating)

               max_user = max(max_user, user)
               max_item = max(max_item, item)

               if rating >= self.threshold:
                  if user not in user_pos:
                     user_pos[user] = set()
                  user_pos[user].add(item)

      self.num_users = max_user + 1
      self.num_items = max_item + 1

      # make sure every user exists in the dict, even if no positives remain
      for user in range(self.num_users):
         if user not in user_pos:
               user_pos[user] = set()

      # -------------------------------------------------
      # 2. Split each user's positives into 70/15/15
      # -------------------------------------------------
      train_pos = {}
      val_pos = {}
      test_pos = {}

      for user in range(self.num_users):
         items = list(user_pos[user])
         random.shuffle(items)

         n = len(items)
         n_train = int(n * 0.7)
         n_val = int(n * 0.15)
         n_test = n - n_train - n_val

         # simple safeguard if positives are very few after thresholding
         if n >= 3:
               if n_train == 0:
                  n_train = 1
               if n_val == 0:
                  n_val = 1
               n_test = n - n_train - n_val
               if n_test <= 0:
                  n_test = 1
                  n_train = max(1, n - n_val - n_test)

         train_items = items[:n_train]
         val_items = items[n_train:n_train + n_val]
         test_items = items[n_train + n_val:]

         train_pos[user] = train_items
         val_pos[user] = val_items
         test_pos[user] = test_items

      # -------------------------------------------------
      # 3. Store train/val/test structures
      # -------------------------------------------------
      for user in range(self.num_users):
         self.train_matrix[user] = set(train_pos[user])
         self.val_ground_truth[user] = set(val_pos[user])
         self.test_ground_truth[user] = set(test_pos[user])

         for item in val_pos[user]:
               self.val_data.append((user, item))

         for item in test_pos[user]:
               self.test_data.append((user, item))

      # -------------------------------------------------
      # 4. Build training data with negative sampling 
      # -------------------------------------------------
      # Resample negatives every epoch for more diverse negative examples.
      self.resample_negatives()
   

   def resample_negatives(self):
      """Resample training negatives (call every epoch)."""

      self.train_data = []

      for user in range(self.num_users):

         pos_items = self.train_matrix[user]

         if len(pos_items) == 0:
            continue

         # all positives (train + val + test)
         all_pos = (
            self.train_matrix[user] |
            self.val_ground_truth[user] |
            self.test_ground_truth[user]
         )

         for item in pos_items:

            # positive
            self.train_data.append((user, item, 1.0))

            # negatives
            for _ in range(self.num_neg):

                  neg_item = random.randint(0, self.num_items - 1)

                  while neg_item in all_pos:
                     neg_item = random.randint(0, self.num_items - 1)

                  self.train_data.append((user, neg_item, 0.0))

      random.shuffle(self.train_data)

   def __len__(self):
      return len(self.train_data)

   def __getitem__(self, idx):
      user, item, label = self.train_data[idx]
      return (
         torch.tensor(user, dtype=torch.long),
         torch.tensor(item, dtype=torch.long),
         torch.tensor(label, dtype=torch.float32),
        )


if __name__ == "__main__":
   # Statistics and sanity checks:

   data = NCFDataset(path="Data/ratings.dat", num_neg=4, threshold=4, seed=42)

   print("=== Dataset statistics ===")
   print("num_users:", data.num_users)
   print("num_items:", data.num_items)

   # Count positives only (ignore sampled negatives)
   train_pos = sum(len(items) for items in data.train_matrix.values())
   val_pos = sum(len(items) for items in data.val_ground_truth.values())
   test_pos = sum(len(items) for items in data.test_ground_truth.values())

   total_pos = train_pos + val_pos + test_pos

   print("\n=== Positive interaction split ===")
   print("train positives:", train_pos)
   print("validation positives:", val_pos)
   print("test positives:", test_pos)
   print("total positives:", total_pos)


   pos_per_user = [
      len(data.train_matrix[u]) +
      len(data.val_ground_truth[u]) +
      len(data.test_ground_truth[u])
      for u in range(data.num_users)
   ]

   print("\n=== Positives per user statistics ===")
   print("mean:", round(statistics.mean(pos_per_user), 2))
   print("median:", statistics.median(pos_per_user))
   print("min:", min(pos_per_user))
   print("max:", max(pos_per_user))

   # Ratios are not exactly 70-15-15, because the split is done per user so there is some rounding.
   # e.g. user with 5 positives: 3 train, 1 val, 1 test (60-20-20 instead of 70-15-15)
   print("\n=== Split ratios ===")
   print("train ratio:", round(train_pos / total_pos, 3))
   print("val ratio:", round(val_pos / total_pos, 3))
   print("test ratio:", round(test_pos / total_pos, 3))

   print("\ntotal training samples (after negative sampling):", len(data.train_data))

   example_user = random.choice(range(data.num_users))
   print("\nExample user:", example_user)
   print("train items:", sorted(list(data.train_matrix[example_user])))
   print("val items:", sorted(list(data.val_ground_truth[example_user])))
   print("test items:", sorted(list(data.test_ground_truth[example_user])))