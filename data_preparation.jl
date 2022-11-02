using Titanic
using CSV
using DataFrames
using ScikitLearn
using VegaLite
using ScikitLearn.CrossValidation: train_test_split
@sk_import preprocessing: StandardScaler
@sk_import preprocessing: OneHotEncoder
@sk_import preprocessing: LabelBinarizer

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
df_train_X, df_test_X = df_train[:,[:Name, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare ]], df_test[:,[:Name, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare ]]
df_train_y= df_train[:, :Survived]

# split train dataset into train + test dataset
X_train, X_test, y_train, y_test = train_test_split(Matrix(df_train_X), df_train_y, test_size=0.2)
# assign back column names
X_train, X_test = rename!(DataFrame(X_train, :auto), names(df_train_X)), rename!(DataFrame(X_test, :auto), names(df_test_X))

function feature_encoding(X)
    X = dropmissing(X, Not([:Age])) 
    # Name: OneHotEncoding Title
    titles = Titanic.title_from_name(X.Name)
    title_resh = reshape(titles, length(titles), 1)
    enc = OneHotEncoder(sparse=false)
    title_ohe = DataFrame(enc.fit_transform(title_resh), convert(Vector{String}, enc.get_feature_names_out(["title"])))
    X = hcat(X, title_ohe)
    # Fare: Scaling
    fare_resh = reshape(X.Fare, length(X.Fare), 1)
    scaler = StandardScaler() # wrong scaling? assumes data is normally distributed which is not the case - maybe min-max normalization? 
    X.Fare = vec(scaler.fit_transform(fare_resh))
    # Age: Get missing Age values by interpolating Means from Pclass and Sex
    new_df = Titanic.age_fill(X)
    X.Age= log10.(new_df.Age)
    # Sex: LabelBinarizer
    lb = LabelBinarizer()
    X.Sex = vec(lb.fit_transform(X[!, :Sex]))
    
    X_final = select!(X, Not([:Name]))
    return X_final
end

X_train_enc, X_test_enc = feature_encoding(X_train), feature_encoding(X_test)
df_test_enc = feature_encoding(df_test_X)

CSV.write("./data/X_train_enc.csv", X_train_enc)
CSV.write("./data/X_test_enc.csv", X_test_enc)
CSV.write("./data/df_test_enc.csv", df_test_enc)
