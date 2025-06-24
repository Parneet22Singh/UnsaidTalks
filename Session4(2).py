import kagglehub
path = kagglehub.dataset_download("anthonypino/melbourne-housing-market")
print("Path to dataset files:", path)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
# Step 1: Check nulls in each column
print("Missing values per column:\n", df.isnull().sum())

# Step 2: Visualize the missing values using a heatmap
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# Step 3: Drop rows where 'Car' is null
df=df[~df['Car'].isnull()]
#df = df[df['Car'].notnull()]

# Step 4: Fill missing 'Car' values with median
df['Car'] = df['Car'].fillna(df['Car'].median())

# Step 5: Fill missing 'BuildingArea' values with mean
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].mean())

# Step 6: Fill missing 'YearBuilt' with most frequent value (mode)
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].mode()[0])

# Step 7: Safely drop 'CouncilArea' if it exists
df.drop(columns=['CouncilArea'], inplace=True, errors='ignore')

# Step 8: Fill all remaining numeric nulls with column mean
df = df.fillna(df.mean(numeric_only=True))

# Step 9: Print rows where more than 2 columns have nulls
rows_with_many_nulls = df[df.isnull().sum(axis=1) > 2]
print("Rows with >2 nulls:\n", rows_with_many_nulls)

# Step 10: Replace missing values in 'BuildingArea' with 0
df['BuildingArea'] = df['BuildingArea'].fillna(0)

# Step 11: Create indicator for missing 'BuildingArea' (before filling)
df['BuildingArea_missing'] = df['BuildingArea'].isnull().astype(int)

# Step 12: Fill missing 'Car' values with random integers between 1 and 4
missing_car_idx = df[df['Car'].isnull()].index
df.loc[missing_car_idx, 'Car'] = np.random.randint(1, 5, size=len(missing_car_idx))

# Step 13: Use KNNImputer for numeric features
numeric_cols = df.select_dtypes(include=['number']).columns
knn = KNNImputer(n_neighbors=3)
df[numeric_cols] = knn.fit_transform(df[numeric_cols])

# Step 14: Create summary of missing data after imputation
missing_summary = df.isnull().mean() * 100
missing_summary = missing_summary[missing_summary > 0]
print("Missing Data Percentage Summary:\n", missing_summary)

# Step 15: Save cleaned data
df.to_csv("cleaned_housing_data.csv", index=False)
print("✅ Cleaned data saved as cleaned_housing_data.csv")

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df = pd.read_csv("cleaned_housing_data.csv")
# Step 16: Identify categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("Categorical Columns:", cat_cols)

# Step 17–18: Label encode 'Type', 'Method', 'SellerG' if they exist
label_cols = ['Type', 'Method', 'SellerG']
for col in label_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])

  from sklearn.preprocessing import OneHotEncoder

# Step 19: One-hot encode 'Regionname'
if 'Regionname' in df.columns:
    ohe = OneHotEncoder(drop='first', sparse_output=False)  # Use this instead of sparse=False
    region_encoded = ohe.fit_transform(df[['Regionname']])
    region_df = pd.DataFrame(region_encoded, columns=ohe.get_feature_names_out(['Regionname']))
    df = pd.concat([df.drop('Regionname', axis=1), region_df], axis=1)


# Step 20: get_dummies for all remaining object columns (like Address, Suburb if still there)
df = pd.get_dummies(df, drop_first=True)

#Step 21:Drop the original categorical columns after encoding
df.drop(columns=cat_cols, inplace=True)

# Step 22: Convert 'Date' into 'Year' and 'Month'
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df.drop('Date', axis=1, inplace=True)

# Step 23:Map Type to custom values: {'h': 0, 'u': 1, 't': 2} 
df['Type_mapped'] = df['Type'].map({'h': 0, 'u': 1, 't': 2})

#Step 24: Use ColumnTransformer to encode multiple categorical features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
ct = ColumnTransformer([
    ('onehot', OneHotEncoder(), ['Regionname', 'Type']),
    ('label', LabelEncoder(), ['Method'])  # Note: LabelEncoder doesn't directly work in ColumnTransformer
])


#Step 25:Convert CouncilArea to category and use .cat.codes
df['CouncilArea'] = df['CouncilArea'].astype('category')
df['CouncilArea_code'] = df['CouncilArea'].cat.codes

# Step 26: Frequency encoding for 'SellerG'
if 'SellerG' in df.columns:
    freq_map = df['SellerG'].value_counts().to_dict()
    df['SellerG_freq'] = df['SellerG'].map(freq_map)

# Step 27: Group rare values of 'SellerG' into 'Other' and encode
if 'SellerG' in df.columns:
    top5 = df['SellerG'].value_counts().nlargest(5).index
df['SellerG'] = df['SellerG'].astype(str)  # Ensure all are strings
df['SellerG_grouped'] = df['SellerG'].apply(lambda x: x if x in top5 else 'Other')
df['SellerG_grouped'] = LabelEncoder().fit_transform(df['SellerG_grouped'])

# Step 28: Utility function to label encode any column
def encode_label_column(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    return df

# Step 29: Target encoding of 'Suburb' based on mean 'Price'
if 'Suburb' in df.columns and 'Price' in df.columns:
    suburb_map = df.groupby('Suburb')['Price'].mean().to_dict()
    df['Suburb_encoded'] = df['Suburb'].map(suburb_map)
    df.drop('Suburb', axis=1, inplace=True)

#Step 30: Save the encoded features into a new Dataframe
df_encoded = df.copy()
df_encoded.to_csv("encoded_housing_data.csv", index=False)
print("✅ Final encoded data saved as encoded_housing_data.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Step 31: List all numerical features
num_features = df.select_dtypes(include=np.number).columns.tolist()
print(f"Numerical features: {num_features}")

# Step 32: Define which features to scale with which scaler
standard_scale_features = ['Distance', 'Landsize', 'BuildingArea']
minmax_scale_features = ['Price', 'Rooms']

# Step 33: Apply StandardScaler to Distance, Landsize, BuildingArea
standard_scaler = StandardScaler()
df_standard_scaled = df.copy()
df_standard_scaled[standard_scale_features] = standard_scaler.fit_transform(df[standard_scale_features])

# Step 34: Apply MinMaxScaler to Price and Rooms
minmax_scaler = MinMaxScaler()
df_minmax_scaled = df_standard_scaled.copy()
df_minmax_scaled[minmax_scale_features] = minmax_scaler.fit_transform(df[minmax_scale_features])

# Step 35: Compare distribution of scaled vs unscaled features using plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(standard_scale_features + minmax_scale_features, 1):
    plt.subplot(3, 3, i)
    sns.kdeplot(df[col], label=f'Original {col}')
    sns.kdeplot(df_minmax_scaled[col], label=f'Scaled {col}')
    plt.legend()
    plt.tight_layout()
plt.show()

# Step 36: Use RobustScaler for features with outliers (Distance, Landsize, BuildingArea)
robust_scaler = RobustScaler()
df_robust_scaled = df.copy()
df_robust_scaled[standard_scale_features] = robust_scaler.fit_transform(df[standard_scale_features])

# Step 37: Fit and transform selected features using ColumnTransformer
col_transformer = ColumnTransformer(
    transformers=[
        ('standard', StandardScaler(), standard_scale_features),
        ('minmax', MinMaxScaler(), minmax_scale_features)
    ],
    remainder='passthrough'  # keep other columns untouched
)
df_col_transformed_array = col_transformer.fit_transform(df)
# Note: ColumnTransformer returns np.ndarray, so reconstruct DataFrame:
new_columns = standard_scale_features + minmax_scale_features + [col for col in df.columns if col not in standard_scale_features + minmax_scale_features]
df_col_transformed = pd.DataFrame(df_col_transformed_array, columns=new_columns)
print("Data after ColumnTransformer:")
print(df_col_transformed.head())

# Step 38: Save scaled DataFrame (example path, update as needed)
df_col_transformed.to_csv('scaled_data.csv', index=False)

# Step 39: Apply PowerTransformer to normalize skewed numeric features (e.g. Landsize, BuildingArea)
power_transformer = PowerTransformer(method='yeo-johnson')  # works with zero and negative values too
skewed_features = ['Landsize', 'BuildingArea']

df_power_scaled = df.copy()
df_power_scaled[skewed_features] = power_transformer.fit_transform(df[skewed_features])

# Step 40: Create histogram of scaled features
df_power_scaled[skewed_features].hist(bins=15, figsize=(10,4))
plt.suptitle('Histogram of Power Transformed Features')
plt.show()

# Step 41: Write reusable function to apply any scaler to a set of columns
def apply_scaler(df, scaler, columns):
    df_copy = df.copy()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy

# Example usage:
df_scaled = apply_scaler(df, StandardScaler(), ['Distance', 'Landsize'])
print(df_scaled.head())

# Step 42: Define features X and target y
X = df.drop(columns=['Price'])
y = df['Price']

# Step 43: Split the dataset into train and test (80-20) with stratification on Regionname
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=X['Regionname'], 
    random_state=42
)

# Step 44: Print shapes of train and test sets
print(f"Train X shape: {X_train.shape}, Train y shape: {y_train.shape}")
print(f"Test X shape: {X_test.shape}, Test y shape: {y_test.shape}")

# Step 45: Visualize price distribution in training and test sets
plt.figure(figsize=(10,4))
sns.kdeplot(y_train, label='Train Price')
sns.kdeplot(y_test, label='Test Price')
plt.legend()
plt.title('Price Distribution: Train vs Test')
plt.show()

# Step 46: Create preprocessing pipeline with ColumnTransformer and Pipeline
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('standard', StandardScaler(), ['Distance', 'Landsize', 'BuildingArea']),
        ('minmax', MinMaxScaler(), ['Rooms']),
        ('passthrough', 'passthrough', ['Regionname'])  # Leave Regionname untouched for now
    ]
)
model_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('regressor', LinearRegression())
])

# Step 47: Train Linear Regression model
model_pipeline.fit(X_train, y_train)
# Step 50: Reusable function for train/test split with preprocessing pipeline
def split_and_preprocess(df, target_col, stratify_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=df[stratify_col], 
        random_state=random_state
    )
    
    # Example pipeline (customize transformers based on your needs)
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    X_train_scaled = pipeline.fit_transform(X_train[numeric_features])
    X_test_scaled = pipeline.transform(X_test[numeric_features])
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# Example call:
X_train_scaled, X_test_scaled, y_train, y_test = split_and_preprocess(df, 'Price', 'Regionname')
print("Shapes after reusable split and scale function:", X_train_scaled.shape, X_test_scaled.shape)

# Step 48: Predict and report score on test set
y_pred = model_pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression R2 score on test set: {r2:.4f}")

# Step 49: Save train and test sets as CSVs
train_df = X_train.copy()
train_df['Price'] = y_train
test_df = X_test.copy()
test_df['Price'] = y_test

train_df.to_csv('train_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)

