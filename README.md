Create a Virtual Environment (Optional but Recommended)

python -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  

Using Conda
git clone https://github.com/yourusername/fault-clustering-analysis.git  
cd fault-clustering-analysis  
conda env create -f environment.yml  
conda activate fault_clustering_env  

1. Prepare the Configuration File  
Modify the config.yaml file to specify your parameters: Example  
shapefile_path: "./lineament.shp"  # Path to the input shapefile  
k: 3  # Number of clusters  
R2: 0.95  # Threshold for linearity filtering  
output_directory: "./output"  # Directory to save logs and results  
log_level: "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)  

3. Run the Script  
python fault_clustering.py --config config.yaml

4. Check the Outputs
Logs: JSON-formatted logs are saved in the specified output_directory.  
Clusters: The clustered data is saved as clusters.csv in the output_directory.  
Visualizations: Any generated maps or histograms will also be saved in the output_directory.

*This project is licensed under the MIT License.*


