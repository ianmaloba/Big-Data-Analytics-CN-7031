import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots without display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create output directory for plots
output_dir = Path('./results/hive_visualizations')
output_dir.mkdir(exist_ok=True)

print("=" * 60)
print("HIVE QUERY VISUALIZATION SCRIPT")
print("=" * 60)

# ============================================================================
# QUERY 1: Attack Distribution Analysis
# ============================================================================
print("\n[1/4] Processing Query 1: Attack Distribution Analysis...")

# Read Query 1 data
q1_df = pd.read_csv(
    './results/hive_results/query1/000000_0',
    names=['traffic_type', 'attack_cat', 'count', 'avg_duration', 'avg_src_bytes', 'avg_dst_bytes']
)

# Clean attack categories - remove leading/trailing spaces and handle duplicates
q1_df['attack_cat'] = q1_df['attack_cat'].astype(str).str.strip()
q1_df = q1_df.groupby(['traffic_type', 'attack_cat']).agg({
    'count': 'sum',
    'avg_duration': 'mean',
    'avg_src_bytes': 'mean',
    'avg_dst_bytes': 'mean'
}).reset_index()

# Calculate counts - convert to Python int immediately
normal_count = int(q1_df[q1_df['traffic_type'] == 'Normal']['count'].sum())
attack_count = int(q1_df[q1_df['traffic_type'] == 'Attack']['count'].sum())
total_count = normal_count + attack_count

print(f"   - Loaded {len(q1_df)} records")
print(f"   - Total traffic records: {total_count:,}")

# Visualization 1.1: Attack vs Normal Traffic Distribution (Pie Chart)
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#2ecc71', '#e74c3c']
explode = (0.05, 0)
plt.pie([normal_count, attack_count],
        labels=['Normal Traffic', 'Attack Traffic'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 14, 'weight': 'bold'})
plt.title('Overall Traffic Distribution: Normal vs Attack', fontsize=16, weight='bold', pad=20)
plt.savefig(output_dir / 'q1_traffic_distribution_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q1_traffic_distribution_pie.png")

# Visualization 1.2: Attack Categories Distribution (Bar Chart)
fig, ax = plt.subplots(figsize=(14, 8))
attack_data = q1_df[q1_df['traffic_type'] == 'Attack'].copy()
attack_data = attack_data.sort_values('count', ascending=True)

bars = ax.barh(attack_data['attack_cat'], attack_data['count'], color='#e74c3c', alpha=0.8)
ax.set_xlabel('Number of Attacks', fontsize=13, weight='bold')
ax.set_ylabel('Attack Category', fontsize=13, weight='bold')
ax.set_title('Distribution of Attack Categories', fontsize=16, weight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2,
            f'{int(width):,}',
            ha='left', va='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'q1_attack_categories_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q1_attack_categories_bar.png")

# Visualization 1.3: Average Bytes by Attack Type (Grouped Bar)
fig, ax = plt.subplots(figsize=(14, 8))
attack_data = q1_df[q1_df['traffic_type'] == 'Attack'].copy()
attack_data = attack_data.sort_values('count', ascending=False).head(10)

x = np.arange(len(attack_data))
width = 0.35

bars1 = ax.bar(x - width/2, attack_data['avg_src_bytes'], width, label='Avg Source Bytes', color='#3498db')
bars2 = ax.bar(x + width/2, attack_data['avg_dst_bytes'], width, label='Avg Destination Bytes', color='#e67e22')

ax.set_xlabel('Attack Category', fontsize=13, weight='bold')
ax.set_ylabel('Average Bytes', fontsize=13, weight='bold')
ax.set_title('Average Data Transfer by Attack Category (Top 10)', fontsize=16, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(attack_data['attack_cat'], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'q1_avg_bytes_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q1_avg_bytes_comparison.png")

# ============================================================================
# QUERY 2: Protocol Analysis
# ============================================================================
print("\n[2/4] Processing Query 2: Protocol Analysis...")

# Read Query 2 data
q2_df = pd.read_csv(
    './results/hive_results/query2/000000_0',
    names=['protocol', 'service_type', 'connection_count', 'attack_count', 'attack_percentage', 'avg_total_bytes']
)

# Clean protocol and service type data - remove leading/trailing spaces
q2_df['protocol'] = q2_df['protocol'].astype(str).str.strip()
q2_df['service_type'] = q2_df['service_type'].astype(str).str.strip()
q2_df = q2_df.groupby(['protocol', 'service_type']).agg({
    'connection_count': 'sum',
    'attack_count': 'sum',
    'attack_percentage': 'mean',
    'avg_total_bytes': 'mean'
}).reset_index()

print(f"   - Loaded {len(q2_df)} protocol/service combinations")

# Visualization 2.1: Attack Percentage by Protocol-Service (Horizontal Bar)
fig, ax = plt.subplots(figsize=(12, 10))
q2_sorted = q2_df.sort_values('attack_percentage', ascending=True)

colors_scale = plt.cm.RdYlGn_r(q2_sorted['attack_percentage'] / 100)
bars = ax.barh(range(len(q2_sorted)), q2_sorted['attack_percentage'], color=colors_scale)

labels = [f"{row['protocol']}/{row['service_type']}" for _, row in q2_sorted.iterrows()]
ax.set_yticks(range(len(q2_sorted)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Attack Percentage (%)', fontsize=13, weight='bold')
ax.set_ylabel('Protocol/Service', fontsize=13, weight='bold')
ax.set_title('Attack Percentage by Protocol and Service Type', fontsize=16, weight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (idx, row) in enumerate(q2_sorted.iterrows()):
    ax.text(row['attack_percentage'] + 1, i, f"{row['attack_percentage']:.1f}%",
            va='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'q2_attack_percentage_by_protocol.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q2_attack_percentage_by_protocol.png")

# Visualization 2.2: Connection Count vs Attack Count (Scatter)
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(q2_df['connection_count'], q2_df['attack_count'],
                    s=q2_df['attack_percentage']*10,
                    c=q2_df['attack_percentage'],
                    cmap='Reds', alpha=0.6, edgecolors='black', linewidth=1)

# Add labels for notable points
for _, row in q2_df.iterrows():
    if row['attack_percentage'] > 50 or row['connection_count'] > 500000:
        ax.annotate(f"{row['protocol']}/{row['service_type']}",
                   (row['connection_count'], row['attack_count']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

ax.set_xlabel('Total Connection Count', fontsize=13, weight='bold')
ax.set_ylabel('Attack Count', fontsize=13, weight='bold')
ax.set_title('Connection Count vs Attack Count by Protocol/Service', fontsize=16, weight='bold', pad=20)
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Attack Percentage (%)', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'q2_connection_vs_attack_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q2_connection_vs_attack_scatter.png")

# Visualization 2.3: Top 5 Protocols by Connection Volume (Stacked Bar)
fig, ax = plt.subplots(figsize=(12, 8))
top5 = q2_df.nlargest(5, 'connection_count')

normal_count_q2 = top5['connection_count'] - top5['attack_count']
attack_count_q2 = top5['attack_count']
labels = [f"{row['protocol']}/{row['service_type']}" for _, row in top5.iterrows()]

x = np.arange(len(top5))
width = 0.6

p1 = ax.bar(x, normal_count_q2, width, label='Normal Traffic', color='#2ecc71')
p2 = ax.bar(x, attack_count_q2, width, bottom=normal_count_q2, label='Attack Traffic', color='#e74c3c')

ax.set_ylabel('Connection Count', fontsize=13, weight='bold')
ax.set_xlabel('Protocol/Service', fontsize=13, weight='bold')
ax.set_title('Top 5 Protocol/Service by Connection Volume', fontsize=16, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'q2_top5_protocols_stacked.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q2_top5_protocols_stacked.png")

# ============================================================================
# QUERY 4: Connection State Analysis
# ============================================================================
print("\n[3/4] Processing Query 4: Connection State Analysis...")

# Read Query 4 data
q4_df = pd.read_csv(
    './results/hive_results/query4/000000_0',
    names=['state', 'total_connections', 'avg_src_bytes', 'stddev_src_bytes',
           'min_src_bytes', 'max_src_bytes', 'median_src_bytes', 'avg_duration', 'attack_count']
)

# Clean connection state data - remove leading/trailing spaces
q4_df['state'] = q4_df['state'].astype(str).str.strip()
q4_df = q4_df.groupby('state').agg({
    'total_connections': 'sum',
    'avg_src_bytes': 'mean',
    'stddev_src_bytes': 'mean',
    'min_src_bytes': 'min',
    'max_src_bytes': 'max',
    'median_src_bytes': 'mean',
    'avg_duration': 'mean',
    'attack_count': 'sum'
}).reset_index()

print(f"   - Loaded {len(q4_df)} connection states")

# Visualization 4.1: Attack Count by Connection State
fig, ax = plt.subplots(figsize=(12, 8))
q4_sorted = q4_df.sort_values('attack_count', ascending=True)

colors = plt.cm.Reds(q4_sorted['attack_count'] / q4_sorted['attack_count'].max())
bars = ax.barh(q4_sorted['state'], q4_sorted['attack_count'], color=colors)

ax.set_xlabel('Number of Attacks', fontsize=13, weight='bold')
ax.set_ylabel('Connection State', fontsize=13, weight='bold')
ax.set_title('Attack Distribution by Connection State', fontsize=16, weight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(q4_sorted.iterrows()):
    ax.text(row['attack_count'], i, f" {int(row['attack_count']):,}",
            va='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'q4_attacks_by_state.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q4_attacks_by_state.png")

# Visualization 4.2: Connection State Statistics (Multiple Subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Connection State Statistical Analysis', fontsize=18, weight='bold', y=1.00)

# Subplot 1: Total Connections
ax1 = axes[0, 0]
ax1.bar(q4_df['state'], q4_df['total_connections'], color='#3498db', alpha=0.8)
ax1.set_ylabel('Total Connections', fontsize=11, weight='bold')
ax1.set_xlabel('Connection State', fontsize=11, weight='bold')
ax1.set_title('Total Connections by State', fontsize=13, weight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=0)

# Subplot 2: Average Duration
ax2 = axes[0, 1]
ax2.bar(q4_df['state'], q4_df['avg_duration'], color='#e67e22', alpha=0.8)
ax2.set_ylabel('Average Duration (seconds)', fontsize=11, weight='bold')
ax2.set_xlabel('Connection State', fontsize=11, weight='bold')
ax2.set_title('Average Connection Duration by State', fontsize=13, weight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=0)

# Subplot 3: Average Source Bytes with Error Bars (stddev)
ax3 = axes[1, 0]
ax3.bar(q4_df['state'], q4_df['avg_src_bytes'],
        yerr=q4_df['stddev_src_bytes'], capsize=5,
        color='#9b59b6', alpha=0.8, error_kw={'linewidth': 2})
ax3.set_ylabel('Average Source Bytes', fontsize=11, weight='bold')
ax3.set_xlabel('Connection State', fontsize=11, weight='bold')
ax3.set_title('Avg Source Bytes by State (with Std Dev)', fontsize=13, weight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.tick_params(axis='x', rotation=0)

# Subplot 4: Attack Percentage
ax4 = axes[1, 1]
attack_pct = (q4_df['attack_count'] / q4_df['total_connections'] * 100)
colors = ['#e74c3c' if pct > 10 else '#2ecc71' for pct in attack_pct]
ax4.bar(q4_df['state'], attack_pct, color=colors, alpha=0.8)
ax4.set_ylabel('Attack Percentage (%)', fontsize=11, weight='bold')
ax4.set_xlabel('Connection State', fontsize=11, weight='bold')
ax4.set_title('Attack Percentage by State', fontsize=13, weight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.tick_params(axis='x', rotation=0)

# Add percentage labels
for i, (state, pct) in enumerate(zip(q4_df['state'], attack_pct)):
    ax4.text(i, pct + 1, f'{pct:.1f}%', ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'q4_state_statistics_multi.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q4_state_statistics_multi.png")

# Visualization 4.3: Source Bytes Distribution (Box Plot Style)
fig, ax = plt.subplots(figsize=(12, 8))

states = q4_df['state'].tolist()
positions = range(len(states))

# Create custom "box plot" using min, median, max
for i, (_, row) in enumerate(q4_df.iterrows()):
    # Draw vertical line from min to max
    ax.plot([i, i], [row['min_src_bytes'], row['max_src_bytes']],
            color='gray', linewidth=2, alpha=0.5)

    # Draw box from avg - stddev to avg + stddev
    box_height = row['stddev_src_bytes']
    box_bottom = max(0, row['avg_src_bytes'] - box_height)
    box_top = row['avg_src_bytes'] + box_height

    ax.add_patch(plt.Rectangle((i-0.3, box_bottom), 0.6, box_top - box_bottom,
                               facecolor='skyblue', edgecolor='black', alpha=0.7))

    # Mark median
    ax.plot([i-0.3, i+0.3], [row['median_src_bytes'], row['median_src_bytes']],
            color='red', linewidth=3, label='Median' if i == 0 else '')

    # Mark average
    ax.scatter([i], [row['avg_src_bytes']], color='green', s=100, zorder=5,
              marker='D', label='Average' if i == 0 else '')

ax.set_xticks(positions)
ax.set_xticklabels(states)
ax.set_xlabel('Connection State', fontsize=13, weight='bold')
ax.set_ylabel('Source Bytes', fontsize=13, weight='bold')
ax.set_title('Source Bytes Distribution by Connection State', fontsize=16, weight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'q4_bytes_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: q4_bytes_distribution.png")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n[4/4] Generating Summary Report...")

summary_file = output_dir / 'summary_statistics.txt'
with open(summary_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("HIVE QUERY ANALYSIS - SUMMARY STATISTICS\n")
    f.write("=" * 70 + "\n\n")

    # Query 1 Summary (use the already calculated Python ints)
    normal_pct = (normal_count / total_count * 100)
    attack_pct = (attack_count / total_count * 100)
    most_common_attack = q1_df[q1_df['traffic_type']=='Attack'].iloc[0]

    f.write("QUERY 1: ATTACK DISTRIBUTION ANALYSIS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Records Analyzed: {total_count:,}\n")
    f.write(f"Normal Traffic Records: {normal_count:,} ({normal_pct:.2f}%)\n")
    f.write(f"Attack Traffic Records: {attack_count:,} ({attack_pct:.2f}%)\n")
    f.write(f"Number of Attack Categories: {len(q1_df[q1_df['traffic_type']=='Attack'])}\n")
    f.write(f"Most Common Attack: {most_common_attack['attack_cat']}\n")
    f.write(f"  - Count: {int(most_common_attack['count']):,}\n\n")

    # Query 2 Summary
    highest_attack = q2_df.iloc[0]
    most_common_proto = q2_df.loc[q2_df['connection_count'].idxmax()]

    f.write("QUERY 2: PROTOCOL ANALYSIS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Protocol/Service Combinations Analyzed: {len(q2_df)}\n")
    f.write(f"Highest Attack Percentage: {highest_attack['protocol']}/{highest_attack['service_type']}")
    f.write(f" ({float(highest_attack['attack_percentage']):.2f}%)\n")
    f.write(f"Most Common Protocol/Service: {most_common_proto['protocol']}/{most_common_proto['service_type']}")
    f.write(f" ({int(most_common_proto['connection_count']):,} connections)\n\n")

    # Query 4 Summary
    most_vulnerable = q4_df.iloc[0]
    most_connections_state = q4_df.loc[q4_df['total_connections'].idxmax()]
    avg_duration = float(q4_df['avg_duration'].mean())

    f.write("QUERY 4: CONNECTION STATE ANALYSIS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Connection States Analyzed: {len(q4_df)}\n")
    f.write(f"Most Vulnerable State: {most_vulnerable['state']}")
    f.write(f" ({int(most_vulnerable['attack_count']):,} attacks)\n")
    f.write(f"State with Most Connections: {most_connections_state['state']}")
    f.write(f" ({int(most_connections_state['total_connections']):,} connections)\n")
    f.write(f"Average Connection Duration: {avg_duration:.4f} seconds\n\n")

    f.write("=" * 70 + "\n")
    f.write("All visualizations have been saved to:\n")
    f.write(f"{output_dir}\n")
    f.write("=" * 70 + "\n")

print(f"   ✓ Saved: summary_statistics.txt")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETE!")
print("=" * 60)
print(f"\nAll visualizations saved to: {output_dir}")
print("\nGenerated files:")
print("  • 3 visualizations for Query 1 (Attack Distribution)")
print("  • 3 visualizations for Query 2 (Protocol Analysis)")
print("  • 3 visualizations for Query 4 (State Analysis)")
print("  • 1 summary statistics file")
print("\nTotal: 10 files generated")
print("\nYou can now include these charts in your report!")
print("=" * 60)