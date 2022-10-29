using Titanic
using CSV
using DataFrames
using ScikitLearn
@sk_import preprocessing: StandardScaler
@sk_import preprocessing: OneHotEncoder

df_train = DataFrame(CSV.File("./data/train.csv"))
df_test = DataFrame(CSV.File("./data/test.csv"))

### Data inspection
describe(df_train)

### fix NAs
#TODO

### Feature Engineering
# Name: extract titles
titles = Titanic.title_from_name(df_train.Name)
df_train.title = titles

# Titles: onehotencode
title_resh = reshape(df_train.title, length(df_train.title), 1) # reshape in a one-column Matrix for StandardScaler.
enc = OneHotEncoder(sparse=false)
title_ohe = DataFrame(enc.fit_transform(title_resh), convert(Vector{String}, enc.get_feature_names_out(["title"])))
df_train = hcat(df_train, title_ohe)

# Fare: scaling
fare_resh = reshape(df_train.Fare, length(df_train.Fare), 1) # reshape in a one-column Matrix for StandardScaler.
scaler = StandardScaler()
df_train.fare_norm = vec(scaler.fit_transform(fare_resh))

# Data inspection after preprocessing
describe(df_train)
