from datasets import load_dataset

dataset = load_dataset("Salesforce/wikitext", "wikitext-103", split="train")

with open("data.txt", "w", encoding="utf-8") as f:
    for line in dataset["text"]:
        if line.strip():  
            f.write(line + "\n")



# import os
# from datasets import load_dataset

# os.environ["HF_HOME"] = "/goinfre/hrabh/.cache/huggingface"

# dataset = load_dataset(
#     "Salesforce/wikitext", 
#     "wikitext-103-raw-v1", 
#     split="train"
# )

# max_lines = 200000

# lines_written = 0
# with open("data.txt", "w", encoding="utf-8") as f:
#     for row in dataset:
#         if lines_written >= max_lines:
#             break
#         text = row["text"]
#         if text.strip():
#             f.write(text + "\n")
#             lines_written += 1

# print(f"Done! Written {lines_written} lines to data.txt.")