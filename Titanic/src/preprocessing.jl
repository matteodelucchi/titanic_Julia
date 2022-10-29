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

