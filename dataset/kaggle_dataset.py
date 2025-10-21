import kagglehub

# Download latest version to same directory as this script
path = kagglehub.dataset_download("kaushiksuresh147/customer-segmentation")
            
print("Path to dataset files:", path)