from utils import db_connect
engine = db_connect()

# your code here
#libraries imported 
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the URL of the dataset and the path where to save it   
dataset_url = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"
save_dir = '/workspaces/machine-learning/data/raw'
file_name = 'AB_NYC_2019.csv'
save_path = os.path.join(save_dir, file_name)
file_path = '/workspaces/machine-learning/data/raw/AB_NYC_2019.csv'


# Download the dataset
response = requests.get(dataset_url)
if response.status_code == 200:
    with open(save_path, 'wb') as file:
        file.write(response.content)
    print(f"Dataset downloaded and saved as {save_path}")
else:
    print(f"Failed to download the dataset. Status code: {response.status_code}")



whole_df = pd.read_csv(file_path)


# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Ensure the output is not truncated

print(whole_df.head())
print(whole_df.tail())


# df dimentions  
print(f"Dataset dimensions: {whole_df.shape}")

#displaying the full dataframe 
display(whole_df.head(20))



display(whole_df.tail(20))
