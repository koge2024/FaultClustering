# fault_clustering.py
# -*- coding: utf-8 -*-
"""
Fault Clustering Analysis Script
Optimized with configuration validation, dynamic logging levels, advanced logging, and parallel processing.
"""

import sys
import math
import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from geopy.distance import geodesic
from pyproj import Geod
import folium
from folium.plugins import FeatureGroupSubGroup, PolyLineTextPath
import pygmt
import yaml
import logging
import json
from datetime import datetime
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
from config_model import Config  # 上述で定義したConfigモデル
from typing import Optional

# --- Configuration Parameters ---
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fault Clustering Analysis Script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file."
    )
    return parser.parse_args()

# --- JSON Logging Setup ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno
        }
        return json.dumps(log_record)

def setup_logger(output_dir: str, log_level: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_filename = f"log_FaultClustering_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    log_filepath = os.path.join(output_dir, log_filename)
    
    logger = logging.getLogger("FaultClusteringLogger")
    logger.setLevel(getattr(logging, log_level))
    
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(JsonFormatter())
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # コンソールには常にINFO以上を表示
    console_handler.setFormatter(JsonFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# --- Helper Functions ---
def load_config(config_path: str) -> Config:
    try:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        config = Config(**config_dict)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def check_shapefile_components(shapefile_path, logger):
    """
    Check for required components of a shapefile.
    """
    directory = os.path.dirname(shapefile_path)
    base_filename = os.path.splitext(os.path.basename(shapefile_path))[0]

    required_files = [".shp", ".shx", ".dbf", ".prj"]
    missing_files = [f"{base_filename}{ext}" for ext in required_files if not os.path.exists(os.path.join(directory, f"{base_filename}{ext}"))]
    
    if missing_files:
        logger.warning(f"Missing files for the shapefile: {missing_files}")
    else:
        logger.info("All required shapefile components are present.")

def convert_to_wgs84(gdf, logger):
    """
    Convert GeoDataFrame to WGS84 CRS.
    """
    if gdf.crs is None:
        logger.warning("No CRS found. Assuming CRS as UTM 52R (EPSG:32652).")
        gdf = gdf.set_crs("EPSG:32652")  # Default to UTM 52R

    logger.info(f"Current CRS: {gdf.crs}")
    gdf = gdf.to_crs("EPSG:4326")
    logger.info(f"Converted CRS: {gdf.crs}")
    return gdf

def extract_lon_lat(line, logger):
    """
    Extract longitude and latitude from a LINESTRING geometry.
    """
    try:
        lon, lat = line.xy
        return pd.DataFrame({'Lon': lon, 'Lat': lat})
    except Exception as e:
        logger.error(f"Error extracting lon/lat: {e}")
        return pd.DataFrame()

def calculate_bearing(start_lon, start_lat, end_lon, end_lat, logger):
    """
    Calculate the azimuth (bearing) between two points.
    """
    try:
        azimuth = math.atan2(end_lon - start_lon, end_lat - start_lat)
        azimuth = math.degrees(azimuth)
        return azimuth + 360 if azimuth < 0 else azimuth
    except Exception as e:
        logger.error(f"Error calculating bearing: {e}")
        return None

def calculate_distance(start_lon, start_lat, end_lon, end_lat, logger):
    """
    Calculate geodesic distance between two points.
    """
    try:
        return geodesic((start_lat, start_lon), (end_lat, end_lon)).kilometers
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return None

def remove_curved_lines(df, threshold, logger):
    """
    Filter DataFrame to remove lines that do not meet the linearity threshold.
    """
    try:
        correlation_coefficient = df.groupby('id').apply(
            lambda group: np.corrcoef(group['Lon'], group['Lat'])[0, 1]
        )
        filtered_ids = correlation_coefficient[correlation_coefficient.abs() > threshold].index
        logger.info(f"Filtered {len(df['id'].unique()) - len(filtered_ids)} curved lines.")
        return df[df['id'].isin(filtered_ids)]
    except Exception as e:
        logger.error(f"Error filtering curved lines: {e}")
        return df

@contextmanager
def timer(name: str, logger):
    start_time = time.time()
    logger.info(f"Started: {name}")
    try:
        yield
    finally:
        end_time = time.time()
        logger.info(f"Finished: {name} in {end_time - start_time:.2f} seconds")

def compute_bearing_distance(row, logger):
    try:
        start_lon = row['Lon'].iloc[0]
        start_lat = row['Lat'].iloc[0]
        end_lon = row['Lon'].iloc[-1]
        end_lat = row['Lat'].iloc[-1]
        direction = calculate_bearing(start_lon, start_lat, end_lon, end_lat, logger)
        distance = calculate_distance(start_lon, start_lat, end_lon, end_lat, logger)
        return (row['id'].iloc[0], direction, distance)
    except Exception as e:
        logger.error(f"Error computing bearing/distance for id {row['id'].iloc[0]}: {e}")
        return (row['id'].iloc[0], None, None)

def calculate_direction_distance_parallel(filtered_df, logger):
    results = {}
    grouped = filtered_df.groupby('id')
    
    with ProcessPoolExecutor() as executor:
        future_to_id = {executor.submit(compute_bearing_distance, group, logger): group_id for group_id, group in grouped}
        for future in as_completed(future_to_id):
            group_id = future_to_id[future]
            try:
                id_val, direction, distance = future.result()
                results[id_val] = {'direction': direction, 'distance': distance}
            except Exception as e:
                logger.error(f"Error in future for id {group_id}: {e}")
    
    # Create DataFrame from results
    direction_distance_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'id'})
    return direction_distance_df

# --- Main Workflow ---
def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logger with dynamic log level
    logger = setup_logger(config.output_directory, config.log_level)
    logger.info("Starting Fault Clustering Analysis Script")
    
    # Log Python version and path
    logger.debug(f"Python version: {sys.version}")
    logger.debug(f"Python path: {sys.path}")
    
    with timer("Load Shapefile", logger):
        check_shapefile_components(config.shapefile_path, logger)
        try:
            gdf = gpd.read_file(config.shapefile_path)
            logger.info(f"Loaded shapefile with {len(gdf)} geometries.")
        except Exception as e:
            logger.error(f"Error reading shapefile: {e}")
            sys.exit(1)
    
    with timer("Convert CRS to WGS84", logger):
        gdf = convert_to_wgs84(gdf, logger)
    
    with timer("Extract Coordinates", logger):
        extracted_data = []
        for idx, row in gdf.iterrows():
            line_df = extract_lon_lat(row['geometry'], logger)
            if not line_df.empty:
                line_df['id'] = idx
                extracted_data.append(line_df)
        
        if not extracted_data:
            logger.error("No valid geometries extracted. Exiting.")
            sys.exit(1)
        
        dff = pd.concat(extracted_data, ignore_index=True)
        logger.info(f"Extracted coordinates for {dff['id'].nunique()} lines.")
    
    with timer("Filter Curved Lines", logger):
        filtered_df = remove_curved_lines(dff, config.R2, logger)
        logger.info(f"Filtered DataFrame has {filtered_df['id'].nunique()} lines after linearity filtering.")
    
    with timer("Calculate Direction and Distance", logger):
        direction_distance_df = calculate_direction_distance_parallel(filtered_df, logger)
        # マージ操作
        filtered_df = filtered_df.merge(direction_distance_df, on='id', how='left')
        # NaNを含む行を削除
        filtered_df.dropna(subset=['direction', 'distance'], inplace=True)
        logger.info(f"DataFrame after dropping NaNs has {len(filtered_df)} records.")
    
    with timer("Perform Clustering", logger):
        try:
            features = np.cos(np.radians(filtered_df['direction'])).values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=config.k, random_state=0).fit(features)
            filtered_df['cluster'] = kmeans.labels_
            logger.info(f"Clustering completed with {config.k} clusters.")
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            sys.exit(1)
    
    with timer("Save Clustered Data", logger):
        clusters_filepath = os.path.join(config.output_directory, "clusters.csv")
        try:
            filtered_df.to_csv(clusters_filepath, index=False)
            logger.info(f"Clustered data saved to {clusters_filepath}")
        except Exception as e:
            logger.error(f"Error saving clustered data: {e}")
    
    # 追加のビジュアライゼーションやファイル出力はここに追加できます
    
    logger.info("Fault Clustering Analysis Script completed successfully.")

if __name__ == "__main__":
    main()
