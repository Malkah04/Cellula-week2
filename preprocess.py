# import pandas as pd
# from sklearn.model_selection import train_test_split
# import torch
# from collections import Counter
# import re

# df_train =pd.read_csv("train.csv")
# df_test =pd.read_csv("test.csv")
# df_toxic =pd.read_csv("test_labels.csv")

# df_test = df_test.merge(
#     df_toxic[["id", "toxic"]],
#     on="id",
#     how="left"
# )

# df_test["toxic"].map("")

# print("null" ,df_train.isnull().sum())

# print("dup " ,df_train.duplicated().sum())

# print(df_train["toxic"].value_counts())
# x_train =df_train["comment_text"]
# y_train =df_train["toxic"]

# x_test =df_test["comment_text"]
# y_test =df_test["toxic"]


# def tokenize(text):
#     text = text.lower()
#     tokens = re.findall(
#         r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b|[!?]",
#         text
#     )

#     return tokens

# counter = Counter()

# for text in x_train:
#     counter.update(tokenize(text))

# print("\nUnique words:", len(counter))


# word_to_idx = {
#     "<PAD>": 0,
#     "<UNK>": 1
# }

# for word, count in counter.most_common():
#     word_to_idx[word] = len(word_to_idx)

# print("Total vocabulary size:", len(word_to_idx))



# def text_to_sequence(text):

#     tokens = tokenize(text)

#     return [
#         word_to_idx.get(
#             word,
#             word_to_idx["<UNK>"]
#         )
#         for word in tokens
#     ]




# train_lengths = x_train.apply(
#     lambda text: len(tokenize(text))
# )

# print("\nToken length statistics:")
# print(train_lengths.describe())


# max_len = int(train_lengths.quantile(0.99))

# print("Max length:", max_len)


# def pad_seq(sequence, max_len):

#     if len(sequence) > max_len:
#         return sequence[:max_len]

#     return sequence + [
#         word_to_idx["<PAD>"]
#     ] * (max_len - len(sequence))




# x_train_seq = [
#     pad_seq(
#         text_to_sequence(text),
#         max_len
#     )
#     for text in x_train
# ]

# x_test_seq = [
#     pad_seq(
#         text_to_sequence(text),
#         max_len
#     )
#     for text in x_test
# ]


# x_train_tensor = torch.tensor(
#     x_train_seq,
#     dtype=torch.long
# )

# x_test_tensor = torch.tensor(
#     x_test_seq,
#     dtype=torch.long
# )

# y_train_tensor = torch.tensor(
#     y_train.values,
#     dtype=torch.long
# )

# y_test_tensor = torch.tensor(
#     y_test.values,
#     dtype=torch.long
# )

