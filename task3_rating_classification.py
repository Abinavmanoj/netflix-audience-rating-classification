
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset.csv")

print("Total titles:", len(df))
print("Number of rating classes:", df["rating"].nunique())

print("\nRating distribution:")
print(df["rating"].value_counts())

# Plot all rating categories
rating_counts = df["rating"].value_counts()

plt.figure(figsize=(12, 6))
rating_counts.plot(kind="bar")
plt.title("Netflix Content Rating Distribution")
plt.xlabel("Content Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rating_distribution.png", dpi=150)
plt.show()

print("\nChart saved as rating_distribution.png")
