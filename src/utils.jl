# utility functions

function get_first_nz_idx{T<:Real}(x::Array{T}, mask = nothing)
    first_nz_idx = 0
    for i in 1:length(x)
        mask != nothing && !mask[i] && continue
        first_nz_idx = i
        abs(x[i]) > eps() && break
    end
    first_nz_idx
end
