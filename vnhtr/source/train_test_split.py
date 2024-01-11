import pandas as pd
import os

# df = pd.read_csv("augmented_anot.csv").sample(frac=1, random_state=42).reset_index(drop=True)
# df1 = pd.read_csv("augmented_anot.csv").sample(frac=1, random_state=42).reset_index(drop=True)[:50000]
df2 = pd.read_csv("wild_anot.csv")
df3 = pd.read_csv("d1_anot.csv")
df4 = pd.read_csv("digits.csv")

df = pd.concat([df2, df3, df4], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# _df = pd.concat([df1, df[:-3000]], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
# print(len(_df))
# _df.to_csv("finetune_anot.csv", index=False)
df[-3000:].to_csv("test_anot.csv", index=False)

# print(len(os.listdir("WildLine")))

# df3 = pd.read_csv("real_anot.csv")
# df4 = pd.read_csv("single_digit.csv")[:20000]

# _df = pd.concat([df3, df4], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
# print(len(_df))
# _df.to_csv("real_anot.csv", index=False)