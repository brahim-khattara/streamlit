import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Streamlit app title
st.title("Water Potability Data Analysis and Visualization")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Read the file
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data")
    st.dataframe(df)

    # Imputation of missing values
    st.write("### Missing Values Before Imputation")
    missing_values = {column: df[column].isnull().sum() for column in df.columns}
    st.write(missing_values)

    # KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df_imputed['Potability'] = df_imputed['Potability'].round()
    st.write("### Missing Values After KNN Imputation")
    missing_values_after_knn = {column: df_imputed[column].isnull().sum() for column in df_imputed.columns}
    st.write(missing_values_after_knn)

    # Regression imputation for any remaining missing values
    def regression_impute(df, target_column):
        df_with_nan = df[df[target_column].isnull()]
        df_without_nan = df[df[target_column].notnull()]
        if not df_with_nan.empty:
            predictors = df.drop(columns=[target_column]).dropna(axis=1, how='any')
            if predictors.empty:
                raise ValueError(f"Not enough columns filled to predict {target_column}.")
            X_train = df_without_nan[predictors.columns]
            y_train = df_without_nan[target_column]
            X_test = df_with_nan[predictors.columns]
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            df.loc[df[target_column].isnull(), target_column] = reg.predict(X_test)
        return df

    for column in df_imputed.columns:
        if df_imputed[column].isnull().sum() > 0:
            df_imputed = regression_impute(df_imputed, column)

    # Data Visualization
    st.write("### Visualizations")
    for column in df_imputed.columns:
        st.write(f"#### {column}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(x=df_imputed[column], ax=axes[0], color='skyblue')
        axes[0].set_title(f'Boxplot of {column}')
        sns.histplot(df_imputed[column], bins=30, kde=True, ax=axes[1], color='green')
        axes[1].set_title(f'Histogram of {column}')
        st.pyplot(fig)

    # Standardization
    scaler = StandardScaler()
    numeric_columns = df_imputed.select_dtypes(include=['float64', 'int64']).columns
    df_imputed[numeric_columns] = scaler.fit_transform(df_imputed[numeric_columns])
    st.write("### Standardized Data")
    st.dataframe(df_imputed.head())

    # Correlation Analysis
    st.write("### Correlation Matrix")
    correlation_matrix = df_imputed.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file to get started.")
