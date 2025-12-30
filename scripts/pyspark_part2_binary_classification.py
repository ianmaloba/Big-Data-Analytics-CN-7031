"""
PySpark Analysis - Part 2: Binary Classification (15 marks)
Task 3.2a: Design and Build a Binary Classifier
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
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
output_dir = Path('./results/pyspark_results')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("PART 2: BINARY CLASSIFICATION (15 MARKS)")
print("Normal vs Attack Detection")
print("=" * 80)

# ============================================================================
# INITIALIZE SPARK
# ============================================================================
print("\n[1/6] Initializing Spark Session...")

spark = SparkSession.builder \
    .appName("Binary Classification") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")
print(f"✓ Spark Version: {spark.version}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[2/6] Loading Dataset...")

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
df = df.na.fill(0)  # Fill nulls

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
# PREPARE FEATURES
# ============================================================================
print("\n[3/6] Preparing Features...")

feature_columns = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
                   'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin',
                   'smeansz', 'dmeansz', 'trans_depth', 'sjit', 'djit',
                   'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
                   'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd']

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features")

# Split data: 70% training, 30% testing
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

train_count = train_data.count()
test_count = test_data.count()

print(f"✓ Training set: {train_count:,} records ({train_count/total_records*100:.1f}%)")
print(f"✓ Test set: {test_count:,} records ({test_count/total_records*100:.1f}%)")
print(f"✓ Number of features: {len(feature_columns)}")

# ============================================================================
# TRAIN RANDOM FOREST CLASSIFIER
# ============================================================================
print("\n[4/6] Training Random Forest Classifier...")

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=100,
    maxDepth=10,
    seed=42
)

pipeline_rf = Pipeline(stages=[assembler, scaler, rf])

print("   Training model (this may take a few minutes)...")
model_rf = pipeline_rf.fit(train_data)
print("✓ Model trained successfully")

# Make predictions
predictions_rf = model_rf.transform(test_data)
predictions_rf.cache()

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n[5/6] Evaluating Model Performance...")

# Metrics
evaluator_auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

auc_score = evaluator_auc.evaluate(predictions_rf)
accuracy = evaluator_acc.evaluate(predictions_rf)
f1_score = evaluator_f1.evaluate(predictions_rf)
precision = evaluator_precision.evaluate(predictions_rf)
recall = evaluator_recall.evaluate(predictions_rf)

print(f"\n--- Random Forest Performance ---")
print(f"✓ Accuracy: {accuracy:.4f}")
print(f"✓ AUC-ROC: {auc_score:.4f}")
print(f"✓ F1-Score: {f1_score:.4f}")
print(f"✓ Precision: {precision:.4f}")
print(f"✓ Recall: {recall:.4f}")

# Get predictions for visualization
pred_labels = predictions_rf.select("label", "prediction", "probability").toPandas()
y_true = pred_labels['label'].values
y_pred = pred_labels['prediction'].values
y_prob = np.array([float(p[1]) for p in pred_labels['probability'].values])

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print(f"\n--- Confusion Matrix ---")
print(f"True Negatives: {cm[0, 0]:,}")
print(f"False Positives: {cm[0, 1]:,}")
print(f"False Negatives: {cm[1, 0]:,}")
print(f"True Positives: {cm[1, 1]:,}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[6/6] Creating Visualizations...")

# Visualization 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'],
            cbar_kws={'label': 'Count'}, ax=ax, annot_kws={'size': 14})

# Add percentages
for i in range(2):
    for j in range(2):
        percent = cm[i, j] / cm.sum() * 100
        ax.text(j + 0.5, i + 0.7, f'({percent:.1f}%)',
                ha='center', va='center', fontsize=11, color='gray')

plt.xlabel('Predicted Label', fontsize=13, weight='bold')
plt.ylabel('True Label', fontsize=13, weight='bold')
plt.title('Binary Classification Confusion Matrix\nRandom Forest Classifier',
          fontsize=15, weight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: binary_confusion_matrix.png")

# Visualization 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
ax.fill_between(fpr, tpr, alpha=0.2, color='#e74c3c')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13, weight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=13, weight='bold')
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=15, weight='bold', pad=20)
ax.legend(loc="lower right", fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'binary_roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: binary_roc_curve.png")

# Visualization 3: Feature Importance
feature_importance = model_rf.stages[-1].featureImportances.toArray()
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

importance_df.to_csv(output_dir / 'binary_feature_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 10))
top_features = importance_df.head(20)
colors = plt.cm.viridis(top_features['Importance'] / top_features['Importance'].max())
bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors, edgecolor='black')

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importance Score', fontsize=13, weight='bold')
ax.set_ylabel('Feature', fontsize=13, weight='bold')
ax.set_title('Top 20 Feature Importance - Binary Classification',
             fontsize=15, weight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['Importance'] + 0.001, i, f"{row['Importance']:.4f}",
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'binary_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: binary_feature_importance.png")

# Visualization 4: Performance Metrics Comparison
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Score': [accuracy, precision, recall, f1_score, auc_score]
}
metrics_df = pd.DataFrame(metrics_data)

fig, ax = plt.subplots(figsize=(10, 6))
colors_map = ['#3498db', '#e67e22', '#9b59b6', '#2ecc71', '#e74c3c']
bars = ax.bar(metrics_df['Metric'], metrics_df['Score'], color=colors_map, alpha=0.8, edgecolor='black')

ax.set_ylabel('Score', fontsize=13, weight='bold')
ax.set_ylim([0, 1.1])
ax.set_title('Binary Classification Performance Metrics', fontsize=15, weight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
           f'{height:.4f}', ha='center', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'binary_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: binary_metrics_comparison.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Save metrics
metrics_results = pd.DataFrame({
    'Metric': ['Accuracy', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall',
               'True Negatives', 'False Positives', 'False Negatives', 'True Positives',
               'Total Test Samples'],
    'Value': [accuracy, auc_score, f1_score, precision, recall,
              int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]), test_count]
})
metrics_results.to_csv(output_dir / 'binary_classification_metrics.csv', index=False)
print("✓ Saved: binary_classification_metrics.csv")

# Classification report
class_report = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
report_df = pd.DataFrame(class_report).transpose()
report_df.to_csv(output_dir / 'binary_classification_report.csv')
print("✓ Saved: binary_classification_report.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PART 2 COMPLETE - BINARY CLASSIFICATION")
print("=" * 80)
print(f"\nModel: Random Forest Classifier")
print(f"Training Samples: {train_count:,}")
print(f"Testing Samples: {test_count:,}")
print(f"Features Used: {len(feature_columns)}")
print(f"\nPerformance Summary:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC-ROC: {auc_score:.4f}")
print(f"  F1-Score: {f1_score:.4f}")
print(f"\nOutput directory: {output_dir}")
print("\nGenerated Files:")
print("  1. binary_confusion_matrix.png")
print("  2. binary_roc_curve.png")
print("  3. binary_feature_importance.png")
print("  4. binary_metrics_comparison.png")
print("  5. binary_classification_metrics.csv")
print("  6. binary_classification_report.csv")
print("  7. binary_feature_importance.csv")
print("\n✓ Total: 7 files generated")
print("=" * 80)

spark.stop()
print("\n✓ Spark session closed")