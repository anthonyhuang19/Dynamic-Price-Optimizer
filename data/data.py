import kagglehub

# Download latest version
path = kagglehub.dataset_download("arashnic/dynamic-pricing-dataset")

print("Path to dataset files:", path)