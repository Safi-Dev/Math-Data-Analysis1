import pandas as pd
from statsmodels.formula import api as smf

# Load the dataset
df = pd.read_csv('data.csv')

# Convert `Vehicle age` to numeric
df['Vehicle age'] = pd.to_numeric(df['Vehicle age'])

# Convert `medical expenses` to numeric
df['medical expenses'] = pd.to_numeric(df['medical expenses'])

# Encode `Crash severity` into dummy variables
df = pd.get_dummies(df, columns=['Crash severity'], prefix=['Crash'])

# Fit the regression model
model = smf.ols('Q("medical expenses") ~ Q("Vehicle age") + Crash_S', data=df).fit()

# Print the model summary
print(model.summary())