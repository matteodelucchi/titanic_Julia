using Titanic
using CSV
using DataFrames
using ScikitLearn
using Flux: onehot
@sk_import preprocessing: StandardScaler
@sk_import preprocessing: OneHotEncoder

df_train = DataFrame(CSV.File("./data/train.csv"))
df_test = DataFrame(CSV.File("./data/test.csv"))
df_all = [df_test; df_train[!, Not(:Survived)]]


### Data inspection
describe(df_train)

### fix NAs
# In train Data
println("Trainings Data")
print(describe(df_train, :nmissing))

# In test data
println("Test Data")
print(describe(df_test, :nmissing))

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

# Age: Means by Class and Sex
df_train = Titanic.age_fill(df_train, df_all)
df_test = Titanic.age_fill(df_test, df_all)
# Check if we have filled the missing
print(describe(df_train, :nmissing))
print(describe(df_test, :nmissing))

# Sex: onehot Encode
df_test.Sex = onehot("female", df_test.Sex)
df_train.Sex = onehot("female", df_train.Sex)


# Data inspection after preprocessing
describe(df_train)
