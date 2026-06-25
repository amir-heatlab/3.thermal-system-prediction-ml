import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    "warmup_time": [12, 13, 14, 16],
    "ambient_temp": [20, 18, None, 10]
}

df = pd.DataFrame(data)

print(df)

#Clean Data
train_df=df.dropna()
#Missing Data
missing_row=df[df["ambient_temp"].isna()]
print(train_df)
print(missing_row)

#Model Creation
model = LinearRegression()

#Model Training
model.fit(
    train_df[["warmup_time"]],
    train_df["ambient_temp"]
)
#Pattern prediction
predicted = model.predict(
    missing_row[["warmup_time"]]
)
#Imputing predicted values into the dataset
df.loc[df["ambient_temp"].isna(), "ambient_temp"] = predicted

print(df)
