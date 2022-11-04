module Titanic

using DataFrames
using Statistics
using VegaLite
using CSV
using ScikitLearn
@sk_import metrics: accuracy_score


include("preprocessing.jl")
include("diagnostics.jl")
include("kaggle.jl")

end # EOM Titanic