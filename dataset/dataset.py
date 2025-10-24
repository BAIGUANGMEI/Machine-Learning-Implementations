import kagglehub


# Download latest version
path = kagglehub.dataset_download("miadul/lifestyle-and-health-risk-prediction")

print("Path to dataset files:", path)
