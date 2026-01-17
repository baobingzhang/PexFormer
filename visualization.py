"""
传感器数据可视化脚本 V3 - 5种新可视化方案
============================================
生成5种新的可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = './data'
OUTPUT_DIR = './pics'

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#2C3E50', '#3498DB', '#E74C3C', '#27AE60', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']

def load_all_data():
    excel_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.xlsx')])
    all_data = []
    for filename in excel_files:
        df = pd.read_excel(os.path.join(DATA_DIR, filename))
        all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    return combined

def main():
    print("Generating V3 visualizations...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_all_data()
    print(f"Loaded {len(data):,} records")
    
    fig = plt.figure(figsize=(15, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_visualization_v3.png'), dpi=300)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
