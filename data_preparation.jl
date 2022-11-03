using Titanic
using CSV
using DataFrames
using ScikitLearn
using VegaLite
using ScikitLearn.CrossValidation: train_test_split
@sk_import preprocessing: MinMaxScaler
@sk_import preprocessing: LabelBinarizer
@sk_import preprocessing: LabelEncoder

df_train = DataFrame(CSV.File("./data/train.csv"))
df_test = DataFrame(CSV.File("./data/test.csv"))

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

### FEATURE ENCODING 

# drop features
df_train_X, df_test_X = df_train[:,[:Survived, :Name, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare ]], df_test[:,[:Name, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare ]]

# split train dataset into train + test dataset
X_train, X_test = train_test_split(Matrix(df_train_X), test_size=0.2)

# assign back column names
X_train, X_test = rename!(DataFrame(X_train, :auto), names(df_train_X)), rename!(DataFrame(X_test, :auto), names(df_train_X))

function feature_encoding(X)
    # Name: LabelEncoding Title
    titles = Titanic.title_from_name(X.Name)
    enc = LabelEncoder()
    X.Name = enc.fit_transform(titles)
    # Fare: Scaling
    X.Fare = replace!(X.Fare, missing => NaN)
    fare_resh = reshape(X.Fare, length(X.Fare), 1)
    scaler = MinMaxScaler() 
    X.Fare = vec(scaler.fit_transform(fare_resh))
    # Age: Get missing Age values by interpolating Means from Pclass and Sex
    new_df = Titanic.age_fill(X)
    X.Age= log10.(new_df.Age)
    # Sex: LabelBinarizer
    lb = LabelBinarizer()
    X.Sex = vec(lb.fit_transform(X[!, :Sex]))
    return X
end

X_train_enc, X_test_enc = feature_encoding(X_train), feature_encoding(X_test)
df_test_enc = feature_encoding(df_test_X)

CSV.write("./data/X_train_enc.csv", X_train_enc)
CSV.write("./data/X_test_enc.csv", X_test_enc)
CSV.write("./data/df_test_enc.csv", df_test_enc)
