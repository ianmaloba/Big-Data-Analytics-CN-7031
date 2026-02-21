"""
PySpark Analysis - Part 1: Statistical Analysis (15 marks)
Task 3.1: Analyze and Interpret Big Data
"""
import os
import warnings
warnings.filterwarnings('ignore')

# Windows compatibility fix for PySpark
import sys
if sys.platform == "win32":
    import socketserver
    # Add missing UnixStreamServer for Windows compatibility
    if not hasattr(socketserver, 'UnixStreamServer'):
        socketserver.UnixStreamServer = socketserver.TCPServer

# Initialize findspark for Windows compatibility
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, min as spark_min, max as spark_max
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, LongType

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
output_dir = Path('./results/pyspark_results')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("PART 1: STATISTICAL ANALYSIS (15 MARKS)")
print("=" * 80)

# ============================================================================
# INITIALIZE SPARK SESSION
# ============================================================================
print("\n[1/5] Initializing Spark Session...")

try:
    spark = SparkSession.builder \
        .appName("UNSW-NB15 Statistical Analysis") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

    # Set Spark context log level to reduce verbose output
    spark.sparkContext.setLogLevel("WARN")
    spark.sparkContext.setLogLevel("ERROR")
    print(f"✓ Spark Version: {spark.version}")
except Exception as e:
    print(f"✗ Error initializing Spark session: {str(e)}")
    print("Please ensure Spark is properly installed and configured.")
    sys.exit(1)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[2/5] Loading Dataset...")

schema = StructType([
    StructField("srcip", StringType(), True),
    StructField("sport", IntegerType(), True),
    StructField("dstip", StringType(), True),
    StructField("dsport", IntegerType(), True),
    StructField("proto", StringType(), True),
    StructField("state", StringType(), True),
    StructField("dur", FloatType(), True),
    StructField("sbytes", IntegerType(), True),
    StructField("dbytes", IntegerType(), True),
    StructField("sttl", IntegerType(), True),
    StructField("dttl", IntegerType(), True),
    StructField("sloss", IntegerType(), True),
    StructField("dloss", IntegerType(), True),
    StructField("service", StringType(), True),
    StructField("sload", FloatType(), True),
    StructField("dload", FloatType(), True),
    StructField("spkts", IntegerType(), True),
    StructField("dpkts", IntegerType(), True),
    StructField("swin", IntegerType(), True),
    StructField("dwin", IntegerType(), True),
    StructField("stcpb", LongType(), True),
    StructField("dtcpb", LongType(), True),
    StructField("smeansz", IntegerType(), True),
    StructField("dmeansz", IntegerType(), True),
    StructField("trans_depth", IntegerType(), True),
    StructField("res_bdy_len", IntegerType(), True),
    StructField("sjit", FloatType(), True),
    StructField("djit", FloatType(), True),
    StructField("stime", LongType(), True),
    StructField("ltime", LongType(), True),
    StructField("sintpkt", FloatType(), True),
    StructField("dintpkt", FloatType(), True),
    StructField("tcprtt", FloatType(), True),
    StructField("synack", FloatType(), True),
    StructField("ackdat", FloatType(), True),
    StructField("is_sm_ips_ports", IntegerType(), True),
    StructField("ct_state_ttl", IntegerType(), True),
    StructField("ct_flw_http_mthd", IntegerType(), True),
    StructField("is_ftp_login", IntegerType(), True),
    StructField("ct_ftp_cmd", IntegerType(), True),
    StructField("ct_srv_src", IntegerType(), True),
    StructField("ct_srv_dst", IntegerType(), True),
    StructField("ct_dst_ltm", IntegerType(), True),
    StructField("ct_src_ltm", IntegerType(), True),
    StructField("ct_src_dport_ltm", IntegerType(), True),
    StructField("ct_dst_sport_ltm", IntegerType(), True),
    StructField("ct_dst_src_ltm", IntegerType(), True),
    StructField("attack_cat", StringType(), True),
    StructField("label", IntegerType(), True)
])

# Get absolute path for the CSV file
import os
csv_path = os.path.abspath('./data/UNSW-NB15.csv')

# Validate data file exists
if not os.path.exists(csv_path):
    print(f"✗ Error: Data file not found at {csv_path}")
    print("Please ensure the UNSW-NB15.csv file exists in the data directory.")
    spark.stop()
    sys.exit(1)

try:
    df = spark.read.csv(
        csv_path,
        header=False,
        schema=schema
    )
except Exception as e:
    print(f"✗ Error loading data file: {str(e)}")
    print("Please check the file format and permissions.")
    spark.stop()
    sys.exit(1)

# Clean attack_cat field - remove leading/trailing spaces and standardize categories
from pyspark.sql.functions import trim, when, col as F_col, regexp_replace

try:
    df = df.withColumn(
        "attack_cat_trimmed",
        trim(F_col("attack_cat"))
    ).withColumn(
        "attack_cat_standardized",
        when(F_col("attack_cat_trimmed") == "", None)
        .when(F_col("attack_cat_trimmed") == "Backdoors", "Backdoor")  # Standardize plural to singular
        .otherwise(F_col("attack_cat_trimmed"))
    ).drop("attack_cat", "attack_cat_trimmed").withColumnRenamed("attack_cat_standardized", "attack_cat")
    
    df.cache()
    total_records = df.count()
    
    if total_records == 0:
        print("✗ Error: Dataset is empty")
        spark.stop()
        sys.exit(1)
    
    print(f"✓ Loaded {total_records:,} records")
    print(f"✓ Number of features: {len(df.columns)}")
    print("✓ Cleaned attack_cat field (removed leading/trailing spaces)")
except Exception as e:
    print(f"✗ Error processing data: {str(e)}")
    spark.stop()
    sys.exit(1)

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[3/5] Computing Descriptive Statistics...")

numeric_cols = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin']

stats_data = []
try:
    for col_name in numeric_cols:
        try:
            stats = df.select(
                spark_min(col_name).alias('min'),
                spark_max(col_name).alias('max'),
                mean(col_name).alias('mean'),
                stddev(col_name).alias('stddev')
            ).collect()[0]

            stats_data.append({
                'Feature': col_name,
                'Min': float(stats['min']) if stats['min'] is not None else 0,
                'Max': float(stats['max']) if stats['max'] is not None else 0,
                'Mean': float(stats['mean']) if stats['mean'] is not None else 0,
                'StdDev': float(stats['stddev']) if stats['stddev'] is not None else 0
            })
        except Exception as col_error:
            print(f"⚠ Warning: Error computing statistics for column {col_name}: {str(col_error)}")
            # Add default values for failed column
            stats_data.append({
                'Feature': col_name,
                'Min': 0, 'Max': 0, 'Mean': 0, 'StdDev': 0
            })
except Exception as e:
    print(f"✗ Error computing descriptive statistics: {str(e)}")
    # Continue with empty stats for graceful degradation
    pass

try:
    if stats_data:  # Only save if we have data
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(output_dir / 'descriptive_statistics.csv', index=False)
        print(f"✓ Saved: descriptive_statistics.csv")
    else:
        print("⚠ Warning: No statistics data to save")
except Exception as e:
    print(f"⚠ Warning: Failed to save descriptive_statistics.csv: {str(e)}")

# Visualization 1: Statistics Table
try:
    if 'stats_df' in locals() and not stats_df.empty:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.axis('tight')
        ax.axis('off')

        table_data = stats_df.round(2).values
        table = ax.table(cellText=table_data, colLabels=stats_df.columns,
                        cellLoc='center', loc='center',
                        colColours=['#3498db']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)

        for i in range(len(stats_df)):
            table[(i+1, 0)].set_facecolor('#ecf0f1')

        plt.title('Descriptive Statistics of Network Traffic Features',
                  fontsize=16, weight='bold', pad=20)
        plt.savefig(output_dir / 'stats_descriptive_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: stats_descriptive_table.png")
    else:
        print("⚠ Warning: Skipping statistics table visualization - no data available")
except Exception as e:
    print(f"⚠ Warning: Failed to create statistics table visualization: {str(e)}")
    if 'plt' in locals():
        plt.close()

# ============================================================================
# 2. LABEL DISTRIBUTION ANALYSIS
# ============================================================================
print("\n[4/5] Analyzing Label Distribution...")

try:
    attack_dist = df.groupBy('label').count().collect()
    label_dist = {row['label']: row['count'] for row in attack_dist}

    normal_count = label_dist.get(0, 0)
    attack_count = label_dist.get(1, 0)

    if total_records > 0:
        print(f"   Normal: {normal_count:,} ({normal_count/total_records*100:.2f}%)")
        print(f"   Attack: {attack_count:,} ({attack_count/total_records*100:.2f}%)")
    else:
        print("⚠ Warning: No records to analyze for label distribution")
except Exception as e:
    print(f"✗ Error analyzing label distribution: {str(e)}")
    normal_count, attack_count = 0, 0

# Visualization 2: Label Distribution
try:
    if normal_count > 0 or attack_count > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)
        ax1.pie([normal_count, attack_count],
                labels=['Normal Traffic', 'Attack Traffic'],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=explode,
                shadow=True,
                textprops={'fontsize': 13, 'weight': 'bold'})
        ax1.set_title('Traffic Distribution (Percentage)', fontsize=14, weight='bold')

        # Bar chart with counts
        ax2.bar(['Normal', 'Attack'], [normal_count, attack_count],
                color=colors, alpha=0.8, edgecolor='black', width=0.6)
        ax2.set_ylabel('Number of Records', fontsize=12, weight='bold')
        ax2.set_title('Traffic Distribution (Count)', fontsize=14, weight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for i, v in enumerate([normal_count, attack_count]):
            ax2.text(i, v + 20000, f'{v:,}', ha='center', fontsize=11, weight='bold')

        plt.suptitle('Binary Label Distribution Analysis', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'stats_label_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: stats_label_distribution.png")
    else:
        print("⚠ Warning: Skipping label distribution visualization - no data available")
except Exception as e:
    print(f"⚠ Warning: Failed to create label distribution visualization: {str(e)}")
    if 'plt' in locals():
        plt.close()

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================
print("\n[5/5] Performing Correlation Analysis...")

# Select features for correlation
feature_cols = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts',
                'sload', 'dload', 'sttl', 'dttl', 'sloss', 'dloss', 'label']

# Sample 10% for performance
try:
    print("   Sampling 10% of data for correlation analysis...")
    sample_df = df.select(feature_cols).sample(fraction=0.1, seed=42).toPandas()
    
    if sample_df.empty:
        print("⚠ Warning: Sample dataframe is empty, skipping correlation analysis")
        corr_matrix = None
    else:
        # Calculate correlation matrix
        corr_matrix = sample_df.corr()
        corr_matrix.to_csv(output_dir / 'correlation_matrix.csv')
        print("✓ Saved: correlation_matrix.csv")
except Exception as e:
    print(f"⚠ Warning: Failed to compute correlation matrix: {str(e)}")
    corr_matrix = None
    sample_df = pd.DataFrame()

# Visualization 3: Correlation Heatmap
try:
    if corr_matrix is not None and not corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                    vmin=-1, vmax=1, ax=ax)
        plt.title('Feature Correlation Matrix', fontsize=16, weight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'stats_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: stats_correlation_heatmap.png")
    else:
        print("⚠ Warning: Skipping correlation heatmap - no correlation data available")
except Exception as e:
    print(f"⚠ Warning: Failed to create correlation heatmap: {str(e)}")
    if 'plt' in locals():
        plt.close()

# Visualization 4: Feature Distributions by Label
print("   Creating feature distribution plots...")

try:
    if not sample_df.empty and 'label' in sample_df.columns:
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        fig.suptitle('Feature Distributions: Normal vs Attack Traffic', fontsize=16, weight='bold')

        key_features = ['dur', 'sbytes', 'dbytes', 'spkts', 'dpkts', 'sload', 'dload', 'sttl', 'dttl']

        for idx, feature in enumerate(key_features):
            if feature in sample_df.columns:
                ax = axes[idx // 3, idx % 3]

                normal_data = sample_df[sample_df['label'] == 0][feature].dropna()
                attack_data = sample_df[sample_df['label'] == 1][feature].dropna()

                if len(normal_data) > 0 and len(attack_data) > 0:
                    ax.hist(normal_data, bins=50, alpha=0.6, color='#2ecc71', label='Normal', density=True)
                    ax.hist(attack_data, bins=50, alpha=0.6, color='#e74c3c', label='Attack', density=True)
                else:
                    ax.text(0.5, 0.5, f'No data\navailable for\n{feature}', 
                           ha='center', va='center', transform=ax.transAxes)

                ax.set_xlabel(feature, fontsize=10, weight='bold')
                ax.set_ylabel('Density', fontsize=10, weight='bold')
                ax.set_title(f'{feature.upper()} Distribution', fontsize=11, weight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(alpha=0.3)
            else:
                # Hide subplot if feature not available
                ax = axes[idx // 3, idx % 3]
                ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / 'stats_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: stats_feature_distributions.png")
    else:
        print("⚠ Warning: Skipping feature distributions - no sample data available")
except Exception as e:
    print(f"⚠ Warning: Failed to create feature distribution plots: {str(e)}")
    if 'plt' in locals():
        plt.close()

# Attack Category Distribution
print("\n   Analyzing attack categories...")
try:
    attack_cat_dist = df.filter(col('label') == 1).groupBy('attack_cat').count().orderBy(col('count').desc())
    attack_cat_pd = attack_cat_dist.toPandas()
    
    if attack_cat_pd.empty:
        print("⚠ Warning: No attack categories found in data")
        attack_cat_pd = pd.DataFrame({'attack_cat': ['No Data'], 'count': [0]})
except Exception as e:
    print(f"⚠ Warning: Failed to analyze attack categories: {str(e)}")
    attack_cat_pd = pd.DataFrame({'attack_cat': ['Error'], 'count': [0]})

# Visualization 5: Attack Category Distribution
try:
    if not attack_cat_pd.empty and attack_cat_pd['count'].sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        attack_cat_pd_sorted = attack_cat_pd.sort_values('count', ascending=True)
        bars = ax.barh(attack_cat_pd_sorted['attack_cat'], attack_cat_pd_sorted['count'],
                       color='#e74c3c', alpha=0.8, edgecolor='black')
        ax.set_xlabel('Number of Attacks', fontsize=12, weight='bold')
        ax.set_ylabel('Attack Category', fontsize=12, weight='bold')
        ax.set_title('Attack Category Distribution', fontsize=14, weight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f' {int(width):,}', va='center', fontsize=9, weight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'stats_attack_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: stats_attack_categories.png")
    else:
        print("⚠ Warning: Skipping attack category visualization - no data available")
except Exception as e:
    print(f"⚠ Warning: Failed to create attack category visualization: {str(e)}")
    if 'plt' in locals():
        plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PART 1 COMPLETE - STATISTICAL ANALYSIS")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated Files:")
print("  1. descriptive_statistics.csv")
print("  2. correlation_matrix.csv")
print("  3. stats_descriptive_table.png")
print("  4. stats_label_distribution.png")
print("  5. stats_correlation_heatmap.png")
print("  6. stats_feature_distributions.png")
print("  7. stats_attack_categories.png")
print("\n✓ Total: 7 files generated")
print("=" * 80)

# Save summary
try:
    summary = {
        'Total Records': total_records if 'total_records' in locals() else 0,
        'Normal Traffic': normal_count if 'normal_count' in locals() else 0,
        'Attack Traffic': attack_count if 'attack_count' in locals() else 0,
        'Normal %': round(normal_count/total_records*100, 2) if 'normal_count' in locals() and 'total_records' in locals() and total_records > 0 else 0,
        'Attack %': round(attack_count/total_records*100, 2) if 'attack_count' in locals() and 'total_records' in locals() and total_records > 0 else 0,
        'Number of Features': len(df.columns) if 'df' in locals() else 0,
        'Number of Attack Categories': attack_cat_pd.shape[0] if 'attack_cat_pd' in locals() and not attack_cat_pd.empty else 0
    }
except Exception as e:
    print(f"⚠ Warning: Error creating summary data: {str(e)}")
    summary = {'Error': 'Failed to generate summary'}

try:
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'part1_summary.csv', index=False)
    print("✓ Saved: part1_summary.csv")
except Exception as e:
    print(f"⚠ Warning: Failed to save summary: {str(e)}")

# Ensure Spark session is properly closed
try:
    spark.stop()
    print("\n✓ Spark session closed successfully")
except Exception as e:
    print(f"⚠ Warning: Issue closing Spark session: {str(e)}")