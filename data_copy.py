import pandas as pd

# Load the dataset (comma-delimited CSV)
df = pd.read_csv('Train.csv')

# Create a copy of the original data
df_cat = df.copy()

price_col = 'SalePrice'

# Remember where SalePrice sits so the new column can take its place
price_idx = df_cat.columns.get_loc(price_col)

# Bin SalePrice into 3 categories using quantiles (roughly equal-sized groups)
labels = ['Low Price', 'Medium Price', 'High Price']
price_category = pd.qcut(df_cat[price_col], q=3, labels=labels)

# Drop the original numeric SalePrice column
df_cat = df_cat.drop(columns=[price_col])

# Insert the new categorical column in the same position SalePrice occupied
df_cat.insert(price_idx, 'PriceCategory', price_category)

# Check the result
print(df_cat['PriceCategory'].value_counts())
print(df_cat[['Id', 'PriceCategory']].head())

# Save the new copy
df_cat.to_csv('Train_with_price_category.csv', index=False)