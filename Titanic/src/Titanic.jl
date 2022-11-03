module Titanic

using DataFrames
using Statistics
using VegaLite


include("preprocessing.jl")
include("diagnostics.jl")
include("kaggle.jl")

end # EOM Titanic