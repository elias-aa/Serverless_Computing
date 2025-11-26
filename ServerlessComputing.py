import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from io import StringIO

st.set_page_config(page_title="RetailNova Serverless FinOps Dashboard", layout="wide")

# Title and Introduction
st.title("☁️ RetailNova Serverless FinOps Dashboard")
st.markdown("""
**Analyze and optimize serverless computing costs across AWS Lambda, Azure Functions, and GCP Functions.**  
This dashboard helps identify cost optimization opportunities across 150+ serverless functions.
""")

# Load dataset
@st.cache_data
def load_data():
    # Read the file and fix the unusual quoting
    # Update this path to match where you save the Serverless_Data.csv file
    data_path = 'Serverless_Data.csv'  # Assumes file is in same directory as this script
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    # Remove quotes and carriage returns from each line
    cleaned_lines = []
    for line in lines:
        # Remove surrounding quotes and carriage returns
        cleaned = line.strip().strip('"').replace('\r', '')
        cleaned_lines.append(cleaned)
    
    # Create a proper CSV string
    csv_content = '\n'.join(cleaned_lines)
    
    # Read with pandas
    df = pd.read_csv(StringIO(csv_content))
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    return df

data = load_data()

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.header("🔍 Filters")

# Environment filter
all_environments = sorted(data['Environment'].unique())
selected_environments = st.sidebar.multiselect(
    "Select Environments",
    all_environments,
    default=all_environments
)

# Memory filter
memory_ranges = st.sidebar.slider(
    "Memory Range (MB)",
    min_value=int(data['MemoryMB'].min()),
    max_value=int(data['MemoryMB'].max()),
    value=(int(data['MemoryMB'].min()), int(data['MemoryMB'].max()))
)

# Cost filter
cost_range = st.sidebar.slider(
    "Cost Range (USD)",
    min_value=float(data['CostUSD'].min()),
    max_value=float(data['CostUSD'].max()),
    value=(float(data['CostUSD'].min()), float(data['CostUSD'].max()))
)

# Apply filters
filtered_data = data[
    (data['Environment'].isin(selected_environments)) &
    (data['MemoryMB'] >= memory_ranges[0]) &
    (data['MemoryMB'] <= memory_ranges[1]) &
    (data['CostUSD'] >= cost_range[0]) &
    (data['CostUSD'] <= cost_range[1])
].copy()

# ============================================
# DATASET OVERVIEW
# ============================================
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Functions", len(filtered_data))
with col2:
    st.metric("Total Monthly Cost", f"${filtered_data['CostUSD'].sum():,.2f}")
with col3:
    st.metric("Total Invocations", f"{filtered_data['InvocationsPerMonth'].sum():,.0f}")
with col4:
    st.metric("Avg Cost/Function", f"${filtered_data['CostUSD'].mean():,.2f}")

st.dataframe(filtered_data.head(10), use_container_width=True)

with st.expander("📋 Full Dataset Info & Summary Statistics"):
    st.write("**Summary Statistics:**")
    st.dataframe(filtered_data.describe(), use_container_width=True)
    st.write("**Missing values per column:**")
    st.write(filtered_data.isna().sum())

st.markdown("---")

# ============================================
# EXERCISE 1: TOP COST CONTRIBUTORS
# ============================================
st.header("💰 Exercise 1: Top Cost Contributors")

# Calculate cumulative cost percentage
cost_sorted = filtered_data.sort_values('CostUSD', ascending=False).copy()
cost_sorted['CumulativeCost'] = cost_sorted['CostUSD'].cumsum()
cost_sorted['CumulativePercent'] = (cost_sorted['CumulativeCost'] / cost_sorted['CostUSD'].sum()) * 100

# Find functions contributing to 80% of spend
top_80_percent = cost_sorted[cost_sorted['CumulativePercent'] <= 80]
if len(top_80_percent) == 0:
    top_80_percent = cost_sorted.head(1)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🏆 Functions Contributing 80% of Spend")
    st.write(f"**{len(top_80_percent)} functions** account for 80% of total serverless spend.")
    
    # Display top contributors table
    display_df = top_80_percent[['FunctionName', 'Environment', 'CostUSD', 'CumulativePercent']].copy()
    display_df['CostUSD'] = display_df['CostUSD'].apply(lambda x: f"${x:,.2f}")
    display_df['CumulativePercent'] = display_df['CumulativePercent'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2:
    # Pareto chart
    fig_pareto = go.Figure()
    
    top_20 = cost_sorted.head(20)
    fig_pareto.add_trace(go.Bar(
        x=top_20['FunctionName'],
        y=top_20['CostUSD'],
        name='Cost (USD)',
        marker_color='steelblue'
    ))
    
    fig_pareto.add_trace(go.Scatter(
        x=top_20['FunctionName'],
        y=top_20['CumulativePercent'],
        name='Cumulative %',
        yaxis='y2',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="orange", 
                         annotation_text="80% threshold", yref='y2')
    
    fig_pareto.update_layout(
        title='Top 20 Functions by Cost (Pareto Analysis)',
        xaxis=dict(title='Function Name', tickangle=45),
        yaxis=dict(title='Cost (USD)', side='left'),
        yaxis2=dict(title='Cumulative %', side='right', overlaying='y', range=[0, 105]),
        height=500,
        showlegend=True,
        legend=dict(x=0.7, y=1.1, orientation='h')
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

# Cost vs Invocation Frequency
st.subheader("📈 Cost vs Invocation Frequency")
fig_scatter = px.scatter(
    filtered_data,
    x='InvocationsPerMonth',
    y='CostUSD',
    color='Environment',
    size='MemoryMB',
    hover_name='FunctionName',
    title='Cost vs Invocation Frequency',
    labels={'InvocationsPerMonth': 'Invocations per Month', 'CostUSD': 'Monthly Cost (USD)'},
    log_x=True
)
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ============================================
# EXERCISE 2: MEMORY RIGHT-SIZING
# ============================================
st.header("🧠 Exercise 2: Memory Right-Sizing Analysis")

st.markdown("""
**Goal:** Identify functions where duration is low but memory allocation is high.  
These functions are candidates for memory reduction to save costs.
""")

# Define thresholds
memory_threshold = st.slider("High Memory Threshold (MB)", 256, 4096, 1024, step=128)
duration_threshold = st.slider("Low Duration Threshold (ms)", 50, 2000, 500, step=50)

# Find over-provisioned functions
over_provisioned = filtered_data[
    (filtered_data['MemoryMB'] >= memory_threshold) &
    (filtered_data['AvgDurationMs'] <= duration_threshold)
].copy()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚠️ Over-Provisioned Functions")
    st.write(f"Found **{len(over_provisioned)}** functions with high memory but low duration.")
    
    if len(over_provisioned) > 0:
        # Calculate potential savings (estimate 50% memory reduction)
        over_provisioned['PotentialSavings'] = over_provisioned['CostUSD'] * 0.3
        total_savings = over_provisioned['PotentialSavings'].sum()
        st.success(f"💵 **Estimated Potential Monthly Savings:** ${total_savings:,.2f}")
        
        display_cols = ['FunctionName', 'Environment', 'MemoryMB', 'AvgDurationMs', 'CostUSD', 'PotentialSavings']
        display_df = over_provisioned[display_cols].copy()
        display_df['CostUSD'] = display_df['CostUSD'].apply(lambda x: f"${x:,.2f}")
        display_df['PotentialSavings'] = display_df['PotentialSavings'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No over-provisioned functions found with current thresholds.")

with col2:
    # Memory vs Duration scatter plot
    fig_memory = px.scatter(
        filtered_data,
        x='AvgDurationMs',
        y='MemoryMB',
        color='CostUSD',
        size='CostUSD',
        hover_name='FunctionName',
        title='Memory vs Duration Analysis',
        labels={'AvgDurationMs': 'Avg Duration (ms)', 'MemoryMB': 'Memory (MB)'},
        color_continuous_scale='RdYlGn_r'
    )
    
    # Add threshold lines
    fig_memory.add_hline(y=memory_threshold, line_dash="dash", line_color="red",
                         annotation_text=f"Memory threshold: {memory_threshold}MB")
    fig_memory.add_vline(x=duration_threshold, line_dash="dash", line_color="blue",
                         annotation_text=f"Duration threshold: {duration_threshold}ms")
    
    # Highlight the over-provisioned zone
    fig_memory.add_shape(
        type="rect",
        x0=0, x1=duration_threshold,
        y0=memory_threshold, y1=filtered_data['MemoryMB'].max() + 200,
        fillcolor="red", opacity=0.1,
        line=dict(width=0)
    )
    
    fig_memory.update_layout(height=500)
    st.plotly_chart(fig_memory, use_container_width=True)

st.markdown("---")

# ============================================
# EXERCISE 3: PROVISIONED CONCURRENCY OPTIMIZATION
# ============================================
st.header("⚡ Exercise 3: Provisioned Concurrency Optimization")

st.markdown("""
**Goal:** Compare cold start rate vs provisioned concurrency cost.  
Decide whether to reduce or remove provisioned concurrency for functions with low cold start rates.
""")

# Filter functions with provisioned concurrency
pc_functions = filtered_data[filtered_data['ProvisionedConcurrency'] > 0].copy()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Provisioned Concurrency Analysis")
    st.write(f"**{len(pc_functions)}** functions have provisioned concurrency enabled.")
    
    if len(pc_functions) > 0:
        # Estimate PC cost (approximate: $0.000004646 per GB-second provisioned)
        pc_functions['EstimatedPCCost'] = pc_functions['ProvisionedConcurrency'] * pc_functions['MemoryMB'] / 1024 * 3600 * 24 * 30 * 0.000004646
        pc_functions['ColdStartRatePercent'] = pc_functions['ColdStartRate'] * 100
        
        # Flag functions where PC might be unnecessary (low cold start, high PC cost)
        pc_functions['Recommendation'] = pc_functions.apply(
            lambda row: '🔴 Consider Removing' if row['ColdStartRate'] < 0.02 and row['ProvisionedConcurrency'] >= 2
            else ('🟡 Review' if row['ColdStartRate'] < 0.05 else '🟢 Keep'),
            axis=1
        )
        
        display_cols = ['FunctionName', 'ProvisionedConcurrency', 'ColdStartRate', 'CostUSD', 'Recommendation']
        st.dataframe(pc_functions[display_cols].sort_values('ProvisionedConcurrency', ascending=False), 
                     use_container_width=True, hide_index=True)
        
        # Summary
        remove_count = len(pc_functions[pc_functions['Recommendation'] == '🔴 Consider Removing'])
        st.warning(f"💡 **{remove_count}** functions may have unnecessary provisioned concurrency.")

with col2:
    if len(pc_functions) > 0:
        # Cold Start Rate vs Provisioned Concurrency
        fig_pc = px.scatter(
            pc_functions,
            x='ColdStartRate',
            y='ProvisionedConcurrency',
            size='CostUSD',
            color='CostUSD',
            hover_name='FunctionName',
            title='Cold Start Rate vs Provisioned Concurrency',
            labels={'ColdStartRate': 'Cold Start Rate', 'ProvisionedConcurrency': 'PC Units'},
            color_continuous_scale='RdYlGn_r'
        )
        
        # Add threshold zone
        fig_pc.add_shape(
            type="rect",
            x0=0, x1=0.02,
            y0=2, y1=pc_functions['ProvisionedConcurrency'].max() + 1,
            fillcolor="red", opacity=0.15,
            line=dict(width=0)
        )
        fig_pc.add_annotation(x=0.01, y=pc_functions['ProvisionedConcurrency'].max(),
                              text="Optimization Zone", showarrow=False, font=dict(color="red"))
        
        fig_pc.update_layout(height=450)
        st.plotly_chart(fig_pc, use_container_width=True)
    else:
        st.info("No functions with provisioned concurrency in the filtered data.")

st.markdown("---")

# ============================================
# EXERCISE 4: UNUSED OR LOW-VALUE WORKLOADS
# ============================================
st.header("🔍 Exercise 4: Detect Unused or Low-Value Workloads")

st.markdown("""
**Goal:** Identify functions with less than 1% of total invocations but disproportionately high cost.
""")

# Calculate invocation and cost percentages
total_invocations = filtered_data['InvocationsPerMonth'].sum()
total_cost = filtered_data['CostUSD'].sum()

analysis_data = filtered_data.copy()
analysis_data['InvocationPercent'] = (analysis_data['InvocationsPerMonth'] / total_invocations) * 100
analysis_data['CostPercent'] = (analysis_data['CostUSD'] / total_cost) * 100
analysis_data['CostEfficiencyRatio'] = analysis_data['CostPercent'] / analysis_data['InvocationPercent'].replace(0, 0.001)

# Low-value workloads: <1% invocations but >1% cost (or high efficiency ratio)
invocation_threshold = st.slider("Invocation Threshold (%)", 0.1, 5.0, 1.0, step=0.1)
cost_threshold = st.slider("Cost Threshold (%)", 0.1, 5.0, 1.0, step=0.1)

low_value = analysis_data[
    (analysis_data['InvocationPercent'] < invocation_threshold) &
    (analysis_data['CostPercent'] > cost_threshold)
].copy()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚠️ Low-Value/Unused Workloads")
    st.write(f"Found **{len(low_value)}** functions with <{invocation_threshold}% invocations but >{cost_threshold}% cost.")
    
    if len(low_value) > 0:
        total_waste = low_value['CostUSD'].sum()
        st.error(f"💸 **Potential Monthly Waste:** ${total_waste:,.2f}")
        
        display_cols = ['FunctionName', 'Environment', 'InvocationsPerMonth', 'InvocationPercent', 'CostUSD', 'CostPercent']
        display_df = low_value[display_cols].sort_values('CostPercent', ascending=False)
        display_df['InvocationPercent'] = display_df['InvocationPercent'].apply(lambda x: f"{x:.3f}%")
        display_df['CostPercent'] = display_df['CostPercent'].apply(lambda x: f"{x:.2f}%")
        display_df['CostUSD'] = display_df['CostUSD'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No low-value workloads detected with current thresholds.")

with col2:
    # Scatter: Invocation % vs Cost %
    fig_value = px.scatter(
        analysis_data,
        x='InvocationPercent',
        y='CostPercent',
        color='Environment',
        size='CostUSD',
        hover_name='FunctionName',
        title='Invocation Share vs Cost Share',
        labels={'InvocationPercent': 'Invocation %', 'CostPercent': 'Cost %'},
        log_x=True,
        log_y=True
    )
    
    # Add diagonal reference line (efficient = 1:1 ratio)
    max_val = max(analysis_data['InvocationPercent'].max(), analysis_data['CostPercent'].max())
    fig_value.add_trace(go.Scatter(
        x=[0.001, max_val],
        y=[0.001, max_val],
        mode='lines',
        name='Efficiency Line (1:1)',
        line=dict(dash='dash', color='gray')
    ))
    
    # Highlight low-value zone
    fig_value.add_shape(
        type="rect",
        x0=0.001, x1=invocation_threshold,
        y0=cost_threshold, y1=100,
        fillcolor="red", opacity=0.1,
        line=dict(width=0)
    )
    
    fig_value.update_layout(height=500)
    st.plotly_chart(fig_value, use_container_width=True)

st.markdown("---")

# ============================================
# EXERCISE 5: COST FORECASTING MODEL
# ============================================
st.header("📈 Exercise 5: Cost Forecasting Model")

st.markdown("""
**Goal:** Build a predictive model for serverless costs using:  
`Cost ≈ Invocations × Duration × Memory × PricingCoefficients + DataTransfer`
""")

# Prepare features for the model
features = ['InvocationsPerMonth', 'AvgDurationMs', 'MemoryMB', 'GBSeconds', 'DataTransferGB', 'ProvisionedConcurrency']
X = filtered_data[features].copy()
y = filtered_data['CostUSD'].copy()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Model Performance")
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("R² (Train)", f"{r2_train:.4f}")
    with metrics_col2:
        st.metric("R² (Test)", f"{r2_test:.4f}")
    with metrics_col3:
        st.metric("MAE (Test)", f"${mae_test:.2f}")
    
    # Feature importance
    st.subheader("📊 Feature Coefficients")
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    fig_coef = px.bar(
        coef_df,
        x='Coefficient',
        y='Feature',
        orientation='h',
        title='Feature Importance (Coefficients)',
        color='Coefficient',
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig_coef, use_container_width=True)

with col2:
    # Actual vs Predicted
    st.subheader("🔮 Actual vs Predicted Costs")
    
    all_predictions = model.predict(X)
    comparison_df = pd.DataFrame({
        'Actual': y,
        'Predicted': all_predictions,
        'FunctionName': filtered_data['FunctionName']
    })
    
    fig_pred = px.scatter(
        comparison_df,
        x='Actual',
        y='Predicted',
        hover_name='FunctionName',
        title='Actual vs Predicted Cost',
        labels={'Actual': 'Actual Cost (USD)', 'Predicted': 'Predicted Cost (USD)'}
    )
    
    # Add perfect prediction line
    max_cost = max(y.max(), all_predictions.max())
    fig_pred.add_trace(go.Scatter(
        x=[0, max_cost],
        y=[0, max_cost],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    fig_pred.update_layout(height=450)
    st.plotly_chart(fig_pred, use_container_width=True)

# Cost Calculator
st.subheader("🧮 Cost Prediction Calculator")
st.markdown("Estimate costs for new serverless functions:")

calc_col1, calc_col2, calc_col3 = st.columns(3)
with calc_col1:
    calc_invocations = st.number_input("Invocations/Month", min_value=0, value=100000, step=10000)
    calc_duration = st.number_input("Avg Duration (ms)", min_value=0, value=500, step=50)
with calc_col2:
    calc_memory = st.selectbox("Memory (MB)", [128, 256, 512, 1024, 1536, 2048, 3072, 4096], index=2)
    calc_gb_seconds = st.number_input("GB-Seconds", min_value=0.0, value=10.0, step=1.0)
with calc_col3:
    calc_data_transfer = st.number_input("Data Transfer (GB)", min_value=0.0, value=20.0, step=5.0)
    calc_pc = st.number_input("Provisioned Concurrency", min_value=0, value=0, step=1)

if st.button("📊 Predict Cost"):
    input_data = np.array([[calc_invocations, calc_duration, calc_memory, calc_gb_seconds, calc_data_transfer, calc_pc]])
    predicted_cost = model.predict(input_data)[0]
    st.success(f"💰 **Predicted Monthly Cost:** ${max(0, predicted_cost):,.2f}")

st.markdown("---")

# ============================================
# EXERCISE 6: CONTAINERIZATION CANDIDATES
# ============================================
st.header("🐳 Exercise 6: Containerization Candidates")

st.markdown("""
**Goal:** Identify workloads that would benefit from containerization:  
- Long-running (>3 seconds)  
- High memory (>2GB)  
- Low invocation frequency
""")

# Define thresholds
duration_thresh = st.slider("Duration Threshold (seconds)", 1.0, 30.0, 3.0, step=0.5) * 1000
memory_thresh = st.slider("Memory Threshold (MB)", 512, 4096, 2048, step=256)
invocation_percentile = st.slider("Low Invocation Percentile", 5, 50, 25, step=5)

invocation_thresh = filtered_data['InvocationsPerMonth'].quantile(invocation_percentile / 100)

# Find containerization candidates
container_candidates = filtered_data[
    (filtered_data['AvgDurationMs'] > duration_thresh) |
    ((filtered_data['MemoryMB'] >= memory_thresh) & 
     (filtered_data['InvocationsPerMonth'] < invocation_thresh))
].copy()

# Score candidates
container_candidates['ContainerScore'] = (
    (container_candidates['AvgDurationMs'] / 1000) * 0.4 +  # Duration weight
    (container_candidates['MemoryMB'] / 1024) * 0.3 +  # Memory weight
    (1 - container_candidates['InvocationsPerMonth'] / filtered_data['InvocationsPerMonth'].max()) * 0.3  # Low invocation weight
)

container_candidates = container_candidates.sort_values('ContainerScore', ascending=False)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🐋 Recommended for Containerization")
    st.write(f"Found **{len(container_candidates)}** functions suitable for containerization.")
    
    if len(container_candidates) > 0:
        potential_savings = container_candidates['CostUSD'].sum() * 0.25  # Estimate 25% savings
        st.success(f"💵 **Estimated Savings from Containerization:** ${potential_savings:,.2f}/month")
        
        display_cols = ['FunctionName', 'Environment', 'AvgDurationMs', 'MemoryMB', 'InvocationsPerMonth', 'CostUSD', 'ContainerScore']
        display_df = container_candidates[display_cols].head(15)
        display_df['AvgDurationMs'] = display_df['AvgDurationMs'].apply(lambda x: f"{x:,.0f}ms ({x/1000:.1f}s)")
        display_df['CostUSD'] = display_df['CostUSD'].apply(lambda x: f"${x:,.2f}")
        display_df['ContainerScore'] = display_df['ContainerScore'].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with col2:
    # Visualization
    fig_container = px.scatter(
        filtered_data,
        x='InvocationsPerMonth',
        y='AvgDurationMs',
        color='MemoryMB',
        size='CostUSD',
        hover_name='FunctionName',
        title='Containerization Analysis: Duration vs Invocations',
        labels={'InvocationsPerMonth': 'Invocations/Month', 'AvgDurationMs': 'Avg Duration (ms)'},
        color_continuous_scale='Viridis',
        log_x=True
    )
    
    # Add threshold lines
    fig_container.add_hline(y=duration_thresh, line_dash="dash", line_color="red",
                            annotation_text=f"Duration threshold: {duration_thresh/1000}s")
    fig_container.add_vline(x=invocation_thresh, line_dash="dash", line_color="blue",
                            annotation_text=f"Low invocation threshold")
    
    fig_container.update_layout(height=500)
    st.plotly_chart(fig_container, use_container_width=True)

# Show specific long-running workloads
st.subheader("⏱️ Long-Running ETL/Batch Workloads (>15 seconds)")
long_running = filtered_data[filtered_data['AvgDurationMs'] > 15000].sort_values('AvgDurationMs', ascending=False)
if len(long_running) > 0:
    display_cols = ['FunctionName', 'Environment', 'AvgDurationMs', 'MemoryMB', 'GBSeconds', 'CostUSD']
    display_df = long_running[display_cols].copy()
    display_df['Duration (sec)'] = display_df['AvgDurationMs'] / 1000
    display_df['CostUSD'] = display_df['CostUSD'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_df[['FunctionName', 'Environment', 'Duration (sec)', 'MemoryMB', 'GBSeconds', 'CostUSD']], 
                 use_container_width=True, hide_index=True)
    st.warning("⚡ These long-running functions are prime candidates for migration to containers (ECS/Fargate/Cloud Run).")
else:
    st.info("No long-running workloads (>15s) found in the filtered data.")

st.markdown("---")

# ============================================
# SUMMARY & RECOMMENDATIONS
# ============================================
st.header("📋 Summary & Recommendations")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Cost Optimization Potential")
    
    # Calculate total potential savings
    memory_savings = filtered_data[
        (filtered_data['MemoryMB'] >= 1024) & 
        (filtered_data['AvgDurationMs'] <= 500)
    ]['CostUSD'].sum() * 0.3
    
    pc_savings = filtered_data[
        (filtered_data['ProvisionedConcurrency'] > 0) & 
        (filtered_data['ColdStartRate'] < 0.02)
    ]['CostUSD'].sum() * 0.2
    
    container_savings = filtered_data[
        filtered_data['AvgDurationMs'] > 3000
    ]['CostUSD'].sum() * 0.25
    
    total_savings = memory_savings + pc_savings + container_savings
    
    st.metric("Memory Right-Sizing", f"${memory_savings:,.2f}")
    st.metric("PC Optimization", f"${pc_savings:,.2f}")
    st.metric("Containerization", f"${container_savings:,.2f}")
    st.success(f"**Total Potential Savings:** ${total_savings:,.2f}/month")

with col2:
    st.subheader("📊 Environment Breakdown")
    env_costs = filtered_data.groupby('Environment')['CostUSD'].sum().reset_index()
    fig_env = px.pie(
        env_costs,
        values='CostUSD',
        names='Environment',
        title='Cost by Environment',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_env, use_container_width=True)

with col3:
    st.subheader("🎯 Key Actions")
    st.markdown("""
    1. **Right-size memory** for low-duration functions
    2. **Remove unnecessary** provisioned concurrency
    3. **Clean up** dev/staging environments
    4. **Containerize** long-running ETL workloads
    5. **Archive** unused functions
    6. **Implement** cost tagging
    """)

# Download filtered analysis
st.subheader("📥 Export Data")
st.download_button(
    label="📥 Download Analysis Report as CSV",
    data=filtered_data.to_csv(index=False).encode("utf-8"),
    file_name="serverless_cost_analysis.csv",
    mime="text/csv"
)

st.success("✅ Analysis complete! Use the sidebar filters to explore specific functions or environments.")