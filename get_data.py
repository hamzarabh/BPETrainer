from datasets import load_dataset

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

with open("data.txt", "w", encoding="utf-8") as f:
    for line in dataset["text"]:
        if line.strip():  
            f.write(line + "\n")