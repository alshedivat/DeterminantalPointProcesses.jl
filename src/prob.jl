"""
Probability functions for DPP and k-DPP.

Methods:
--------
    pmf(dpp::DeterminantalPointProcess, z::Array{Int})
    logpmf(dpp::DeterminantalPointProcess, z::Array{Int})
    pmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    logpmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
"""

"""Compute the log probability of a sample `z` under the given DPP.
"""
function logpmf(dpp::DeterminantalPointProcess, z::Array{Int})
    L_z_eigvals = eigvals(dpp.L[z, z])
    return sum(log.(L_z_eigvals)) - sum(log.(dpp.Lfact.values .+ 1))
end

"""Compute the log probability of a sample `z` under the given k-DPP.
"""
function logpmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    L_z_eigvals = eigvals(dpp.L[z, z])
    return sum(log.(L_z_eigvals)) .- log(elem_symm_poly(dpp.Lfact.values, k)[end, end])
end

"""Compute the probability of a sample `z` under the given DPP.
"""
function pmf(dpp::DeterminantalPointProcess, z::Array{Int})
    exp(logpmf(dpp, z))
end

"""Compute the probability of a sample `z` under the given DPP.
"""
function pmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    exp(logpmf(dpp, z, k))
end
