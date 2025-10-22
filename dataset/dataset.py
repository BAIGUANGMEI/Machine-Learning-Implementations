import kagglehub

# Download latest version to same directory as this script
path = kagglehub.dataset_download("andonians/random-linear-regression")
            
print("Path to dataset files:", path)