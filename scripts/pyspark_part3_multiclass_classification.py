"""
PySpark Analysis - Part 3: Multi-class Classification (20 marks)
Task 3.2b: Multi-class Classifier for 10 Categories
(1 Normal + 9 Attack Types)
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
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
output_dir = Path('./results/pyspark_results')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("PART 3: MULTI-CLASS CLASSIFICATION (20 MARKS)")
print("10 Categories: 1 Normal + 9 Attack Types")
print("=" * 80)

# ============================================================================
# INITIALIZE SPARK
# ============================================================================
print("\n[1/7] Initializing Spark Session...")

spark = SparkSession.builder \
    .appName("Multi-class Classification") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"✓ Spark Version: {spark.version}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[2/7] Loading Dataset...")

schema = StructType([
    StructField("srcip", StringType(), True), StructField("sport", IntegerType(), True),
    StructField("dstip", StringType(), True), StructField("dsport", IntegerType(), True),
    StructField("proto", StringType(), True), StructField("state", StringType(), True),
    StructField("dur", FloatType(), True), StructField("sbytes", IntegerType(), True),
    StructField("dbytes", IntegerType(), True), StructField("sttl", IntegerType(), True),
    StructField("dttl", IntegerType(), True), StructField("sloss", IntegerType(), True),
    StructField("dloss", IntegerType(), True), StructField("service", StringType(), True),
    StructField("sload", FloatType(), True), StructField("dload", FloatType(), True),
    StructField("spkts", IntegerType(), True), StructField("dpkts", IntegerType(), True),
    StructField("swin", IntegerType(), True), StructField("dwin", IntegerType(), True),
    StructField("stcpb", LongType(), True), StructField("dtcpb", LongType(), True),
    StructField("smeansz", IntegerType(), True), StructField("dmeansz", IntegerType(), True),
    StructField("trans_depth", IntegerType(), True), StructField("res_bdy_len", IntegerType(), True),
    StructField("sjit", FloatType(), True), StructField("djit", FloatType(), True),
    StructField("stime", LongType(), True), StructField("ltime", LongType(), True),
    StructField("sintpkt", FloatType(), True), StructField("dintpkt", FloatType(), True),
    StructField("tcprtt", FloatType(), True), StructField("synack", FloatType(), True),
    StructField("ackdat", FloatType(), True), StructField("is_sm_ips_ports", IntegerType(), True),
    StructField("ct_state_ttl", IntegerType(), True), StructField("ct_flw_http_mthd", IntegerType(), True),
    StructField("is_ftp_login", IntegerType(), True), StructField("ct_ftp_cmd", IntegerType(), True),
    StructField("ct_srv_src", IntegerType(), True), StructField("ct_srv_dst", IntegerType(), True),
    StructField("ct_dst_ltm", IntegerType(), True), StructField("ct_src_ltm", IntegerType(), True),
    StructField("ct_src_dport_ltm", IntegerType(), True), StructField("ct_dst_sport_ltm", IntegerType(), True),
    StructField("ct_dst_src_ltm", IntegerType(), True), StructField("attack_cat", StringType(), True),
    StructField("label", IntegerType(), True)
])

df = spark.read.csv('file:///' + os.path.abspath('./data/UNSW-NB15.csv'), header=False, schema=schema)
df = df.na.fill(0)

# Clean attack_cat field - remove leading/trailing spaces and handle empty values
from pyspark.sql.functions import trim, when, col as F_col
df = df.withColumn(
    "attack_cat",
    when(trim(F_col("attack_cat")) == "", None).otherwise(trim(F_col("attack_cat")))
)

total_records = df.count()
print(f"✓ Loaded {total_records:,} records")
print("✓ Cleaned attack_cat field (removed leading/trailing spaces)")

# ============================================================================
# PREPARE MULTI-CLASS LABELS
# ============================================================================
print("\n[3/7] Preparing Multi-class Labels...")

# Map attack categories (label=0 becomes "Normal", others keep their attack_cat)
df_multiclass = df.withColumn(
    "attack_category",
    when(col("label") == 0, "Normal").otherwise(col("attack_cat"))
)

# Check class distribution
print("\n--- Class Distribution ---")
class_dist = df_multiclass.groupBy("attack_category").count().orderBy(col("count").desc())
class_dist_pd = class_dist.toPandas()
print(class_dist_pd.to_string(index=False))

# Visualization 1: Class Distribution
fig, ax = plt.subplots(figsize=(12, 8))
class_dist_sorted = class_dist_pd.sort_values('count', ascending=True)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(class_dist_sorted)))
bars = ax.barh(class_dist_sorted['attack_category'], class_dist_sorted['count'],
               color=colors, edgecolor='black')

ax.set_xlabel('Number of Samples', fontsize=13, weight='bold')
ax.set_ylabel('Attack Category', fontsize=13, weight='bold')
ax.set_title('Multi-class Dataset Distribution\n(1 Normal + 9 Attack Types)',
             fontsize=15, weight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f' {int(width):,}', va='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'multiclass_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Saved: multiclass_distribution.png")

# Index string labels to numeric
indexer = StringIndexer(inputCol="attack_category", outputCol="class_label")
df_indexed = indexer.fit(df_multiclass).transform(df_multiclass)

# Get class mapping
label_to_class = indexer.fit(df_multiclass).labels
num_classes = len(label_to_class)
print(f"\n✓ Number of classes: {num_classes}")
print(f"✓ Classes: {', '.join(label_to_class)}")

# ============================================================================
# PREPARE FEATURES
# ============================================================================
print("\n[4/7] Preparing Features...")

feature_columns = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                   'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin',
                   'smeansz', 'dmeansz', 'trans_depth', 'sjit', 'djit',
                   'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
                   'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd']

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# Split data
train_data, test_data = df_indexed.randomSplit([0.7, 0.3], seed=42)

train_count = train_data.count()
test_count = test_data.count()

print(f"✓ Training set: {train_count:,} records")
print(f"✓ Test set: {test_count:,} records")

# ============================================================================
# TRAIN RANDOM FOREST (MULTI-CLASS)
# ============================================================================
print("\n[5/7] Training Random Forest Classifier...")

rf_multi = RandomForestClassifier(
    featuresCol="features",
    labelCol="class_label",
    numTrees=100,
    maxDepth=10,
    seed=42
)

pipeline_rf = Pipeline(stages=[assembler, scaler, rf_multi])

print("   Training model (this may take several minutes)...")
model_rf = pipeline_rf.fit(train_data)
print("✓ Model trained successfully")

# Make predictions
predictions_rf = model_rf.transform(test_data)
predictions_rf.cache()

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n[6/7] Evaluating Model Performance...")

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="class_label", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="class_label", predictionCol="prediction", metricName="f1"
)
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="class_label", predictionCol="prediction", metricName="weightedPrecision"
)
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="class_label", predictionCol="prediction", metricName="weightedRecall"
)

accuracy = evaluator_acc.evaluate(predictions_rf)
f1_score = evaluator_f1.evaluate(predictions_rf)
precision = evaluator_precision.evaluate(predictions_rf)
recall = evaluator_recall.evaluate(predictions_rf)

print(f"\n--- Random Forest Performance ---")
print(f"✓ Accuracy: {accuracy:.4f}")
print(f"✓ Weighted F1-Score: {f1_score:.4f}")
print(f"✓ Weighted Precision: {precision:.4f}")
print(f"✓ Weighted Recall: {recall:.4f}")

# Get predictions for sklearn metrics
pred_data = predictions_rf.select("class_label", "prediction", "attack_category").toPandas()
y_true = pred_data['class_label'].values.astype(int)
y_pred = pred_data['prediction'].values.astype(int)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[7/7] Creating Visualizations...")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Visualization 2: Confusion Matrix
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=label_to_class,
            yticklabels=label_to_class,
            cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 9})

plt.xlabel('Predicted Class', fontsize=13, weight='bold')
plt.ylabel('True Class', fontsize=13, weight='bold')
plt.title('Multi-class Classification Confusion Matrix\n(10 Categories)',
          fontsize=15, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: multiclass_confusion_matrix.png")

# Classification Report
class_report = classification_report(y_true, y_pred,
                                    target_names=label_to_class,
                                    output_dict=True,
                                    zero_division=0)

report_df = pd.DataFrame(class_report).transpose()
report_df.to_csv(output_dir / 'multiclass_classification_report.csv')
print("✓ Saved: multiclass_classification_report.csv")

# Visualization 3: Per-Class Performance Metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

metrics_to_plot = ['precision', 'recall', 'f1-score']
colors_map = ['#3498db', '#e67e22', '#9b59b6']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]

    # Get metric values for each class
    values = []
    for cls in label_to_class:
        if cls in class_report:
            values.append(class_report[cls][metric])
        else:
            values.append(0)

    colors = plt.cm.viridis(np.array(values))
    bars = ax.bar(range(len(label_to_class)), values,
                   color=colors_map[idx], alpha=0.8, edgecolor='black')

    ax.set_xticks(range(len(label_to_class)))
    ax.set_xticklabels(label_to_class, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(metric.capitalize(), fontsize=12, weight='bold')
    ax.set_title(f'{metric.capitalize()} per Class', fontsize=13, weight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', fontsize=8, weight='bold')

plt.suptitle('Multi-class Classification Performance Metrics',
             fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'multiclass_performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: multiclass_performance_metrics.png")

# Visualization 4: Normalized Confusion Matrix (Heatmap %)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=label_to_class,
            yticklabels=label_to_class,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax, annot_kws={'size': 9})

plt.xlabel('Predicted Class', fontsize=13, weight='bold')
plt.ylabel('True Class', fontsize=13, weight='bold')
plt.title('Normalized Confusion Matrix (Percentage)\n(10 Categories)',
          fontsize=15, weight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(output_dir / 'multiclass_confusion_normalized.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: multiclass_confusion_normalized.png")

# Visualization 5: Feature Importance
feature_importance = model_rf.stages[-1].featureImportances.toArray()
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

importance_df.to_csv(output_dir / 'multiclass_feature_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 10))
top_features = importance_df.head(20)
colors = plt.cm.plasma(top_features['Importance'] / top_features['Importance'].max())
bars = ax.barh(range(len(top_features)), top_features['Importance'],
               color=colors, edgecolor='black')

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importance Score', fontsize=13, weight='bold')
ax.set_ylabel('Feature', fontsize=13, weight='bold')
ax.set_title('Top 20 Feature Importance - Multi-class Classification',
             fontsize=15, weight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['Importance'] + 0.001, i, f"{row['Importance']:.4f}",
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'multiclass_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: multiclass_feature_importance.png")

# Visualization 6: Overall Metrics Bar Chart
overall_metrics = {
    'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score'],
    'Score': [accuracy, precision, recall, f1_score]
}
metrics_df = pd.DataFrame(overall_metrics)

fig, ax = plt.subplots(figsize=(10, 6))
colors_map = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71']
bars = ax.bar(metrics_df['Metric'], metrics_df['Score'],
              color=colors_map, alpha=0.8, edgecolor='black')

ax.set_ylabel('Score', fontsize=13, weight='bold')
ax.set_ylim([0, 1.1])
ax.set_title('Multi-class Classification Overall Performance',
             fontsize=15, weight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{height:.4f}', ha='center', fontsize=11, weight='bold')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(output_dir / 'multiclass_overall_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: multiclass_overall_metrics.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Save overall metrics
metrics_results = pd.DataFrame({
    'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score',
               'Number of Classes', 'Training Samples', 'Testing Samples'],
    'Value': [accuracy, precision, recall, f1_score, num_classes, train_count, test_count]
})
metrics_results.to_csv(output_dir / 'multiclass_metrics.csv', index=False)
print("✓ Saved: multiclass_metrics.csv")

# Save class distribution
class_dist_pd.to_csv(output_dir / 'multiclass_class_distribution.csv', index=False)
print("✓ Saved: multiclass_class_distribution.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PART 3 COMPLETE - MULTI-CLASS CLASSIFICATION")
print("=" * 80)
print(f"\nModel: Random Forest Classifier")
print(f"Number of Classes: {num_classes}")
print(f"Classes: {', '.join(label_to_class[:3])}... (and {num_classes-3} more)")
print(f"Training Samples: {train_count:,}")
print(f"Testing Samples: {test_count:,}")
print(f"Features Used: {len(feature_columns)}")
print(f"\nPerformance Summary:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Weighted F1-Score: {f1_score:.4f}")
print(f"  Weighted Precision: {precision:.4f}")
print(f"  Weighted Recall: {recall:.4f}")
print(f"\nOutput directory: {output_dir}")
print("\nGenerated Files:")
print("  1. multiclass_distribution.png")
print("  2. multiclass_confusion_matrix.png")
print("  3. multiclass_confusion_normalized.png")
print("  4. multiclass_performance_metrics.png")
print("  5. multiclass_feature_importance.png")
print("  6. multiclass_overall_metrics.png")
print("  7. multiclass_classification_report.csv")
print("  8. multiclass_feature_importance.csv")
print("  9. multiclass_metrics.csv")
print(" 10. multiclass_class_distribution.csv")
print("\n✓ Total: 10 files generated")
print("=" * 80)

spark.stop()
print("\n✓ Spark session closed")