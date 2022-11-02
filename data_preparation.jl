using Titanic
using CSV
using DataFrames
using ScikitLearn
using VegaLite
using ScikitLearn.CrossValidation: train_test_split
using Flux: onehot
@sk_import preprocessing: StandardScaler
@sk_import preprocessing: OneHotEncoder

df_train = DataFrame(CSV.File("./data/train.csv"))
df_test = DataFrame(CSV.File("./data/test.csv"))
df_all = [df_test; df_train[!, Not(:Survived)]]


# drop missing values + split into train/test

#train_cleaned = dropmissing(df_train[!, [2,3,5,6,7,8,10]])

### Data inspection
describe(df_train)

#visualize
SibSp_Survival= @vlplot(data=df_train)+
@vlplot(:bar, x={:SibSp, bin=true}, y="count()", color={:Survived, type = "nominal"})

Parch_Survival= @vlplot(data=df_train)+
@vlplot(:bar, x={:Parch, bin=true}, y="count()", color={:Survived, type = "nominal"})

Age_Survival= @vlplot(data=df_train)+
@vlplot(:bar, x={:Age, bin=true}, y="count()", color={:Survived, type = "nominal"})

Fare_Survival= @vlplot(data=df_train)+
@vlplot(:bar, x={:Fare, bin={step=20}}, y="count()", color={:Survived, type = "nominal"})

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

# Tickets: separate letters from numbers of train tickets
# train_tickets = Array{String}(undef, size(df_train.Ticket, 1), 2)
# for (i, t) in enumerate(df_train.Ticket)
#     if ' ' in collect(t)
#         split_tickets = split.(t, " ")
#         if length(split_tickets) == 2
#             train_tickets[i,:] = split_tickets
#         else
#             train_tickets[i,:] = [join([split_tickets[1], split_tickets[2]], " "), split_tickets[3]]
#         end
#     else
#         train_tickets[i, 2] = t
#     end
#     return train_tickets
# end

# Fare: scaling
fare_resh = reshape(df_train.Fare, length(df_train.Fare), 1) # reshape in a one-column Matrix for StandardScaler.
scaler = StandardScaler()
df_train.fare_norm = vec(scaler.fit_transform(fare_resh))

# Age: Means by Class and Sex
df_train = Titanic.age_fill(df_train)
df_test = Titanic.age_fill(df_test)
# Check if we have filled the missing
print(describe(df_train, :nmissing))
print(describe(df_test, :nmissing))
# Normalize Age data
df_test.Age = log10.(df_test.Age)
df_train.Age = log10.(df_train.Age)

# Sex: onehot Encode
df_test.Sex = onehot("female", df_test.Sex)
df_train.Sex = onehot("female", df_train.Sex)


# Data inspection after preprocessing
describe(df_train)

# save
df_train_cleaned = df_train[:, Not(["PassengerId", "Ticket", "Name", "Embarked"])]
CSV.write("./data/train_cleaned.csv", df_train_cleaned)
CSV.write("./data/test_cleaned.csv", df_test)

# X = train_cleaned[!, 2:7]
# y = train_cleaned[!, 1]
#X_train, X_test, y_train, y_test = train_test_split(Array(X), y, test_size=0.2)

