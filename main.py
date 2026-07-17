import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import plotly.express as px


# Set page configuration for a wide layout
st.set_page_config(page_title="Advanced EDA & Cleaning Tool", layout="wide")


# Main Title
st.title("Advanced Automated EDA & Data Cleaning Application")
st.write("Upload your dataset below to visualize distributions, manage outliers, clean data, and export the results.")


# ---------------------------------------------------------
# COLLAPSIBLE TUTORIAL & APPLICATION INFO (Main Panel)
# ---------------------------------------------------------
with st.expander("📖 Quick Tutorial & Features Guide (Click to Expand)", expanded=False):
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        ### What This App Provides:
        * **Automated EDA:** Instantly view dataset dimensions, data types, missing value percentages, and descriptive statistics.
        * **Interactive Visualizations:** Analyze column distributions and feature correlations dynamically.
        * **Data Cleaning:** Handle missing data, eliminate duplicates, and filter outliers automatically.
        * **Flexible Export:** Download your refined dataset in both CSV and Excel formats, plus a cleaning report.
        """)

    with col_t2:
        st.markdown("""
        ### How to Use the App:
        1. **Upload:** Click the browse button below to upload a `.csv`, `.xlsx`, or `.xls` file.
        2. **Inspect:** Review the **Exploratory Data Analysis** section to understand data quality and distributions.
        3. **Configure:** Use the control panel in Section 2 to select your data cleaning strategies.
        4. **Apply & Download:** Click "Apply Cleaning", then navigate to Section 3 to export your clean data and cleaning report.
        """)


st.divider()


# =============================================================
# HELPER FUNCTIONS (modular, reusable)
# =============================================================

def compute_health_score(dataframe: pd.DataFrame) -> tuple[int, str]:
    """
    Compute a Dataset Health Score from 0-100 based on:
    - Missing value ratio      (up to -40 points)
    - Duplicate row ratio      (up to -20 points)
    - Constant column ratio    (up to -20 points)
    - High outlier percentage  (up to -20 points, IQR-based on numeric cols)
    Returns (score, status_label).
    """
    score = 100
    total_cells = dataframe.shape[0] * dataframe.shape[1]

    # Missing values penalty (max -40)
    if total_cells > 0:
        missing_ratio = dataframe.isna().sum().sum() / total_cells
        score -= int(min(missing_ratio * 200, 40))

    # Duplicate rows penalty (max -20)
    if dataframe.shape[0] > 0:
        dup_ratio = dataframe.duplicated().sum() / dataframe.shape[0]
        score -= int(min(dup_ratio * 100, 20))

    # Constant columns penalty (max -20)
    if dataframe.shape[1] > 0:
        constant_ratio = sum(dataframe[c].nunique(dropna=False) <= 1 for c in dataframe.columns) / dataframe.shape[1]
        score -= int(min(constant_ratio * 100, 20))

    # Outlier percentage penalty (max -20, IQR on numeric cols)
    numeric_cols = dataframe.select_dtypes(include="number").columns.tolist()
    if numeric_cols and dataframe.shape[0] > 0:
        outlier_flags = []
        for col in numeric_cols:
            q1 = dataframe[col].quantile(0.25)
            q3 = dataframe[col].quantile(0.75)
            iqr = q3 - q1
            outlier_flags.append(((dataframe[col] < q1 - 1.5 * iqr) | (dataframe[col] > q3 + 1.5 * iqr)).sum())
        total_outliers = sum(outlier_flags)
        total_numeric_cells = dataframe.shape[0] * len(numeric_cols)
        outlier_ratio = total_outliers / total_numeric_cells if total_numeric_cells > 0 else 0
        score -= int(min(outlier_ratio * 100, 20))

    score = max(0, min(100, score))

    if score >= 90:
        status = "✅ Excellent"
    elif score >= 75:
        status = "🟢 Good"
    elif score >= 50:
        status = "🟡 Needs Cleaning"
    else:
        status = "🔴 Poor"

    return score, status


def classify_columns(dataframe: pd.DataFrame) -> dict:
    """Return counts of each column type: Numeric, Categorical, Boolean, Datetime, Object."""
    counts = {"Numeric": 0, "Categorical": 0, "Boolean": 0, "Datetime": 0, "Object": 0}
    for col in dataframe.columns:
        dtype = dataframe[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            counts["Boolean"] += 1
        elif pd.api.types.is_numeric_dtype(dtype):
            counts["Numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            counts["Datetime"] += 1
        elif isinstance(dtype, pd.CategoricalDtype):
            counts["Categorical"] += 1
        else:
            counts["Object"] += 1
    return counts


def memory_usage_mb(dataframe: pd.DataFrame) -> float:
    """Return approximate memory usage in MB."""
    return dataframe.memory_usage(deep=True).sum() / (1024 ** 2)


def missing_value_bar_chart(dataframe: pd.DataFrame):
    """Return a Plotly horizontal bar chart for columns with missing values, sorted descending."""
    missing = dataframe.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        return None
    fig = px.bar(
        x=missing.values,
        y=missing.index.tolist(),
        orientation="h",
        labels={"x": "Missing Count", "y": "Column"},
        title="Missing Values by Column",
        color=missing.values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(coloraxis_showscale=False, height=max(250, len(missing) * 35))
    return fig


def apply_cleaning_operations(
    df_source: pd.DataFrame,
    columns_to_drop: list,
    clean_option: str,
    constant_impute_value,
    remove_dup: bool,
    duplicate_cols: list,
    duplicate_strategy: str,
    handle_outliers: bool,
    outlier_method: str | None,
    iqr_multiplier: float,
    z_threshold: float,
    cleaning_log: dict,
) -> pd.DataFrame:
    """Apply all cleaning operations and update the cleaning_log in place. Returns cleaned df."""
    df_out = df_source.copy()

    # Drop columns
    if columns_to_drop:
        df_out = df_out.drop(columns=columns_to_drop)
        cleaning_log["operations"].append({"operation": "drop_columns", "columns": columns_to_drop})

    numeric_cols_cleaned = df_out.select_dtypes(include=["number"]).columns.tolist()

    # Missing value handling
    if clean_option == "Drop rows containing missing values":
        before_na = df_out.shape[0]
        df_out = df_out.dropna()
        cleaning_log["operations"].append({
            "operation": "drop_na",
            "rows_removed": before_na - df_out.shape[0],
        })

    elif clean_option == "Impute values (Median for numeric, Mode for categorical)":
        cleaning_log_op = {"operation": "impute_median_mode", "details": []}
        for col in df_out.columns:
            if pd.api.types.is_numeric_dtype(df_out[col]):
                median_val = df_out[col].median()
                if not pd.isna(median_val):
                    df_out[col] = df_out[col].fillna(median_val)
                    cleaning_log_op["details"].append(f"{col}: median={median_val}")
            else:
                if not df_out[col].mode().empty:
                    mode_val = df_out[col].mode()[0]
                    df_out[col] = df_out[col].fillna(mode_val)
                    cleaning_log_op["details"].append(f"{col}: mode={mode_val}")
        cleaning_log["operations"].append(cleaning_log_op)

    elif clean_option == "Impute numeric with mean, categorical with mode":
        cleaning_log_op = {"operation": "impute_mean_mode", "details": []}
        for col in df_out.columns:
            if pd.api.types.is_numeric_dtype(df_out[col]):
                mean_val = df_out[col].mean()
                if not pd.isna(mean_val):
                    df_out[col] = df_out[col].fillna(mean_val)
                    cleaning_log_op["details"].append(f"{col}: mean={mean_val}")
            else:
                if not df_out[col].mode().empty:
                    mode_val = df_out[col].mode()[0]
                    df_out[col] = df_out[col].fillna(mode_val)
                    cleaning_log_op["details"].append(f"{col}: mode={mode_val}")
        cleaning_log["operations"].append(cleaning_log_op)

    elif clean_option == "Impute with constant value":
        cleaning_log_op = {"operation": "impute_constant", "value": constant_impute_value}
        for col in df_out.columns:
            df_out[col] = df_out[col].fillna(constant_impute_value)
        cleaning_log["operations"].append(cleaning_log_op)

    elif clean_option == "Impute by random sampling from observed values":
        cleaning_log_op = {"operation": "impute_random_sample", "details": []}
        for col in df_out.columns:
            missing_mask = df_out[col].isna()
            n_missing = missing_mask.sum()
            if n_missing == 0:
                continue
            valid_vals = df_out[col].dropna()
            if not valid_vals.empty:
                sampled = valid_vals.sample(n=n_missing, replace=True).values
                df_out.loc[missing_mask, col] = sampled
                cleaning_log_op["details"].append(f"{col}: filled {n_missing} values by random sampling")
        cleaning_log["operations"].append(cleaning_log_op)

    # Duplicates
    if remove_dup:
        before_dup = df_out.shape[0]
        keep_map = {
            "Drop all duplicate rows": False,
            "Keep first occurrence": "first",
            "Keep last occurrence": "last",
        }
        keep_val = keep_map.get(duplicate_strategy, "first")
        if duplicate_cols:
            df_out = df_out.drop_duplicates(subset=duplicate_cols, keep=keep_val)
        else:
            df_out = df_out.drop_duplicates(keep=keep_val)
        cleaning_log["operations"].append({
            "operation": "drop_duplicates",
            "strategy": duplicate_strategy,
            "columns": duplicate_cols if duplicate_cols else "all",
            "rows_removed": before_dup - df_out.shape[0],
        })

    # Outliers
    if handle_outliers and outlier_method and numeric_cols_cleaned:
        outlier_op = {
            "operation": "handle_outliers",
            "method": outlier_method,
            "columns_processed": [],
            "rows_removed": 0,
        }

        if "IQR" in outlier_method:
            flag_only = "flag" in outlier_method
            for col in numeric_cols_cleaned:
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                lb = q1 - iqr_multiplier * iqr
                ub = q3 + iqr_multiplier * iqr
                is_outlier = (df_out[col] < lb) | (df_out[col] > ub)
                if flag_only:
                    df_out[f"is_outlier_{col}"] = is_outlier
                    outlier_op["columns_processed"].append({"column": col, "action": "flagged", "lower_bound": lb, "upper_bound": ub})
                else:
                    before_out = df_out.shape[0]
                    df_out = df_out[(df_out[col] >= lb) & (df_out[col] <= ub)]
                    removed = before_out - df_out.shape[0]
                    outlier_op["columns_processed"].append({"column": col, "action": "removed", "lower_bound": lb, "upper_bound": ub, "rows_removed": removed})
                    outlier_op["rows_removed"] += removed

        else:  # Z-score
            flag_only = "flag" in outlier_method
            for col in numeric_cols_cleaned:
                mean_col = df_out[col].mean()
                std_col = df_out[col].std()
                if std_col == 0:
                    continue
                z_scores = (df_out[col] - mean_col) / std_col
                is_outlier = z_scores.abs() > z_threshold
                if flag_only:
                    df_out[f"is_outlier_{col}"] = is_outlier
                    outlier_op["columns_processed"].append({"column": col, "action": "flagged", "threshold": z_threshold})
                else:
                    before_out = df_out.shape[0]
                    df_out = df_out[~is_outlier]
                    removed = before_out - df_out.shape[0]
                    outlier_op["columns_processed"].append({"column": col, "action": "removed", "threshold": z_threshold, "rows_removed": removed})
                    outlier_op["rows_removed"] += removed

        cleaning_log["operations"].append(outlier_op)

    return df_out


# ---------------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])


if uploaded_file is not None:
    try:
        # Load Data with a Spinner Indicator
        with st.spinner("Reading dataset file... Please wait."):
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

        # Basic validation
        if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
            st.error("The uploaded file does not contain valid tabular data (no rows or columns).")
            st.info("Please verify that the input file contains properly structured tabular data.")
            st.stop()

        st.success("File loaded successfully!")

        # Base copies for processing
        df_original = df.copy()
        df_cleaned = df.copy()

        # ---------------------------------------------------------
        # 1. EDA & VISUALIZATION SECTION
        # ---------------------------------------------------------
        st.header("1. Exploratory Data Analysis & Visualizations")

        # High-level Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Duplicate Rows", df.duplicated().sum())

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

        # ---------------------------------------------------------
        # NEW: Dataset Health Score
        # ---------------------------------------------------------
        st.subheader("📊 Dataset Health Score")
        health_score, health_status = compute_health_score(df)
        h1, h2, h3 = st.columns([1, 1, 4])
        with h1:
            st.metric("Health Score", f"{health_score} / 100")
        with h2:
            st.metric("Status", health_status)
        with h3:
            score_bar = px.bar(
                x=[health_score],
                y=["Health"],
                orientation="h",
                range_x=[0, 100],
                color=[health_score],
                color_continuous_scale=["red", "orange", "green"],
                range_color=[0, 100],
            )
            score_bar.update_layout(
                height=80,
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=False,
                coloraxis_showscale=False,
                xaxis=dict(showticklabels=True),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(score_bar, use_container_width=True, config={"displayModeBar": False})

        # ---------------------------------------------------------
        # NEW: Feature Summary (column type classification)
        # ---------------------------------------------------------
        st.subheader("🗂️ Feature Summary")
        col_types = classify_columns(df)
        ft_cols = st.columns(len(col_types))
        for i, (label, count) in enumerate(col_types.items()):
            with ft_cols[i]:
                st.metric(label, count)

        # ---------------------------------------------------------
        # NEW: Memory Usage
        # ---------------------------------------------------------
        st.subheader("💾 Memory Usage")
        mem1, mem2 = st.columns(2)
        with mem1:
            st.metric("Dataset Size", f"{df.shape[0]:,} rows × {df.shape[1]:,} cols")
        with mem2:
            st.metric("Approx. Memory", f"{memory_usage_mb(df):.2f} MB")

        # Separate Numeric Columns safely
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # ---------------------------------------------------------
        # NEW: Visual Missing Value Summary
        # ---------------------------------------------------------
        st.subheader("🔍 Missing Value Summary")
        missing_chart = missing_value_bar_chart(df)
        if missing_chart:
            st.plotly_chart(missing_chart, use_container_width=True)
        else:
            st.success("✅ No missing values detected in this dataset.")

        # Visualization Sub-section
        if numeric_cols:
            vis_col1, vis_col2 = st.columns(2)

            with vis_col1:
                st.subheader("Data Distribution Analysis")
                selected_box = st.selectbox("Select a numeric column to view distribution:", numeric_cols)
                if selected_box:
                    fig_hist = px.histogram(df, x=selected_box, marginal="box", title=f"Distribution of {selected_box}")
                    st.plotly_chart(fig_hist, use_container_width=True)

            with vis_col2:
                st.subheader("Correlation Heatmap")
                if len(numeric_cols) > 1:
                    # NEW: correlation method selector
                    corr_method = st.selectbox(
                        "Correlation method:",
                        ["pearson", "spearman", "kendall"],
                        key="corr_method",
                    )
                    corr_matrix = df[numeric_cols].corr(method=corr_method)
                    fig_heat = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title=f"Correlation Matrix ({corr_method.capitalize()})",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Not enough numeric columns to generate a correlation matrix.")
        else:
            st.warning("No numeric columns detected in the dataset. Visualizations are limited.")

        # ---------------------------------------------------------
        # NEW: Categorical Analysis
        # ---------------------------------------------------------
        if categorical_cols:
            st.subheader("📋 Categorical Column Analysis")
            selected_cat = st.selectbox("Select a categorical column:", categorical_cols, key="cat_analysis")
            if selected_cat:
                top_n = st.slider("Top N categories to show:", min_value=2, max_value=min(50, df[selected_cat].nunique()), value=min(10, df[selected_cat].nunique()), key="top_n_cat")
                value_counts = df[selected_cat].value_counts().head(top_n).reset_index()
                value_counts.columns = ["Category", "Count"]

                cat_c1, cat_c2 = st.columns([1, 2])
                with cat_c1:
                    st.dataframe(value_counts, use_container_width=True, hide_index=True)
                with cat_c2:
                    fig_cat = px.bar(
                        value_counts,
                        x="Count",
                        y="Category",
                        orientation="h",
                        title=f"Top {top_n} Categories — {selected_cat}",
                        color="Count",
                        color_continuous_scale="Blues",
                    )
                    fig_cat.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_cat, use_container_width=True)

        # ---------------------------------------------------------
        # 2. ADVANCED DATA CLEANING SECTION
        # ---------------------------------------------------------
        st.header("2. Practical Data Cleaning Features")

        # Cleaning strategy summary to show before applying
        cleaning_summary_lines = []

        clean_col1, clean_col2 = st.columns(2)

        with clean_col1:
            st.subheader("Missing Value & Duplicate Management")

            # Duplicate management
            remove_dup = st.checkbox("Automatically remove duplicate rows", value=True)
            duplicate_cols = st.multiselect(
                "Columns to define duplicates (optional, uses all columns if none selected):",
                df.columns.tolist()
            )
            duplicate_strategy = st.selectbox(
                "Duplicate handling strategy:",
                [
                    "Drop all duplicate rows",
                    "Keep first occurrence",
                    "Keep last occurrence"
                ]
            )

            # Missing value management
            clean_option = st.selectbox(
                "Missing value strategy:",
                [
                    "Keep missing values (No imputation)",
                    "Drop rows containing missing values",
                    "Impute values (Median for numeric, Mode for categorical)",
                    "Impute numeric with mean, categorical with mode",
                    "Impute with constant value",
                    "Impute by random sampling from observed values"
                ]
            )

            constant_impute_value = None
            if clean_option == "Impute with constant value":
                constant_impute_value = st.text_input("Constant value for imputation (number or text):")

        with clean_col2:
            st.subheader("Outlier & Column Management")

            # Outlier management
            handle_outliers = st.checkbox("Handle outliers", value=False)

            outlier_method = None
            iqr_multiplier = 1.5
            z_threshold = 2.0

            if handle_outliers:
                outlier_method = st.selectbox(
                    "Outlier handling method:",
                    [
                        "IQR (Interquartile Range) - remove outliers",
                        "IQR (Interquartile Range) - flag outliers",
                        "Z-score - remove outliers",
                        "Z-score - flag outliers"
                    ]
                )

                if "IQR" in outlier_method:
                    iqr_multiplier = st.slider("IQR multiplier (default 1.5):", 1.0, 3.0, 1.5, 0.1)
                else:
                    z_threshold = st.slider("Z-score threshold (default 2.0):", 1.0, 4.0, 2.0, 0.1)

            # Column management
            columns_to_drop = st.multiselect("Select columns to remove completely (Optional):", df.columns.tolist())

        # Build cleaning summary
        if columns_to_drop:
            cleaning_summary_lines.append(f"- Columns to drop: {columns_to_drop}")

        if clean_option != "Keep missing values (No imputation)":
            cleaning_summary_lines.append(f"- Missing value strategy: {clean_option}")
            if clean_option == "Impute with constant value" and constant_impute_value:
                cleaning_summary_lines.append(f"  - Constant value: {constant_impute_value}")

        if remove_dup:
            cleaning_summary_lines.append(f"- Duplicate handling: {duplicate_strategy}")
            if duplicate_cols:
                cleaning_summary_lines.append(f"  - Defined by columns: {duplicate_cols}")

        if handle_outliers and outlier_method:
            cleaning_summary_lines.append(f"- Outlier method: {outlier_method}")
            if "IQR" in outlier_method:
                cleaning_summary_lines.append(f"  - IQR multiplier: {iqr_multiplier}")
            else:
                cleaning_summary_lines.append(f"  - Z-score threshold: {z_threshold}")

        # Show summary if anything is configured
        if cleaning_summary_lines:
            st.subheader("Cleaning Summary (Before Applying)")
            st.info("\n".join(cleaning_summary_lines))

        # Apply Cleaning Button
        apply_cleaning = st.button("Apply Cleaning", type="primary")

        cleaning_log = {
            "original_rows": df_original.shape[0],
            "original_columns": df_original.shape[1],
            "operations": []
        }

        if apply_cleaning:
            with st.spinner("Executing data cleaning routines..."):
                df_cleaned = apply_cleaning_operations(
                    df_source=df_original,
                    columns_to_drop=columns_to_drop,
                    clean_option=clean_option,
                    constant_impute_value=constant_impute_value,
                    remove_dup=remove_dup,
                    duplicate_cols=duplicate_cols,
                    duplicate_strategy=duplicate_strategy,
                    handle_outliers=handle_outliers,
                    outlier_method=outlier_method,
                    iqr_multiplier=iqr_multiplier,
                    z_threshold=z_threshold,
                    cleaning_log=cleaning_log,
                )

        # ---------------------------------------------------------
        # 3. POST-CLEANING SUMMARY & EXPORT
        # ---------------------------------------------------------
        st.header("3. Cleaned Data Summary & Export")

        st.subheader("Post-Cleaned Data Preview")
        st.dataframe(df_cleaned.head(10))

        final_col1, final_col2 = st.columns(2)
        with final_col1:
            st.metric("Final Row Count", df_cleaned.shape[0], delta=int(df_cleaned.shape[0] - df_original.shape[0]))
        with final_col2:
            st.metric("Final Column Count", df_cleaned.shape[1], delta=int(df_cleaned.shape[1] - df_original.shape[1]))

        # Wrap the file conversion buffers in a Spinner since large files take time to serialize
        with st.spinner("Generating download files (CSV, Excel, and cleaning report)..."):
            # Export Buffers
            csv_buffer = io.StringIO()
            df_cleaned.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_cleaned.to_excel(writer, index=False, sheet_name='Cleaned_Data')
            excel_data = excel_buffer.getvalue()

            # Cleaning report buffer
            cleaning_log["final_rows"] = df_cleaned.shape[0]
            cleaning_log["final_columns"] = df_cleaned.shape[1]
            report_json = json.dumps(cleaning_log, indent=2)
            report_buffer = io.StringIO(report_json)
            report_data = report_buffer.getvalue()

        # Download Buttons
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        with dl_col1:
            st.download_button(
                label="⬇️ Download Cleaned Data as CSV",
                data=csv_data,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )
        with dl_col2:
            st.download_button(
                label="⬇️ Download Cleaned Data as Excel",
                data=excel_data,
                file_name="cleaned_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with dl_col3:
            st.download_button(
                label="⬇️ Download Cleaning Report (JSON)",
                data=report_data,
                file_name="cleaning_report.json",
                mime="application/json"
            )

    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        st.info("Please verify that the input file contains properly structured tabular data.")