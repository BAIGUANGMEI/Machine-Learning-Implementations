import kagglehub

# Download latest version to same directory as this script
path = kagglehub.dataset_download("dileep070/heart-disease-prediction-using-logistic-regression")
          
print("Path to dataset files:", path)