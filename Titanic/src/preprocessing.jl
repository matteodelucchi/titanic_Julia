"""
Extract title from names.
Args: DataFrame column with names incl. titles.
Returns: Vector with titles.
"""
function title_from_name(n)
    # init title Vector
    titles = Vector{String}(undef, length(n))
    # extract everything between . and ,
    for i in 1:length(n)
        beforedot = split(n[i], ".")[1]
        aftercoma = split(beforedot, ", ")[2]
        titles[i] = aftercoma
    end # EOI

    return titles
end



"""
Calculate the missing mean values based on the Passenger Class and Sex
Args: DataFrame column with names incl. titles.
Returns: The DataFrame with the calculated means.
"""
function age_fill(data)
    # Calculate the mean based on category and sex over the entire data
    mean_age = sort(combine(groupby(data, [:Pclass, :Sex]), :Age => (x -> mean(skipmissing(x)))))

    for i in 1:size(data, 1)
        if ismissing(data[i, :Age]) # Check if age is missing
            if data[i, :Sex] == "female" # Check for gender and then use the respective mean
                if data[i, :Pclass] == 1
                    data[i, :Age] = mean_age[!, :Age_function][1]
                elseif data[i, :Pclass] == 2
                    data[i, :Age] = mean_age[!, :Age_function][3]
                elseif data[i, :Pclass] == 3
                    data[i, :Age] = mean_age[!, :Age_function][5]
                end
            elseif data[i, :Sex] == "male" # Check for gender and then use the respective mean
                if data[i, :Pclass] == 1
                    data[i, :Age] = mean_age[!, :Age_function][2]
                elseif data[i, :Pclass] == 2
                    data[i, :Age] = mean_age[!, :Age_function][4]
                elseif data[i, :Pclass] == 3
                    data[i, :Age] = mean_age[!, :Age_function][6]
                end
            end
        end
    end
    return data
end