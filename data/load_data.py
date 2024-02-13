# load_data.py

from datasets import load_dataset

def load_and_display_dataset(dataset_name, split_ratio):
    dataset = load_dataset(dataset_name, split=split_ratio)
    print("Dataset shape: ", dataset.shape)
    # Displaying first entry for example
    index = 0
    print("Instruction: \n", dataset["instruction"][index])
    print("Input: \n", dataset["input"][index])
    print("Output: \n", dataset["output"][index])
    print("Text: \n", dataset["text"][index])
    return dataset

if __name__ == "__main__":
    load_and_display_dataset("vicgalle/alpaca-gpt4", "train[:10000]")
