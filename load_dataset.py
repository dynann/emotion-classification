from datasets import load_dataset

# Load the dataset
dataset_name = "stapesai/ssi-speech-emotion-recognition"
print(f"Loading dataset: {dataset_name}...")

try:
    dataset = load_dataset(dataset_name)
    print("Dataset loaded successfully!")
    print(dataset.keys())
    
    # Example: print the first example from the train split if it exists
    if 'train' in dataset:
        print("\nFirst example in 'train' split:")
        print(dataset['train'].features)
except Exception as e:
    print(f"Error loading dataset: {e}")
