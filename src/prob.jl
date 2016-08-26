"""
Probability functions for DPP and k-DPP.

Methods:
--------
    pmf(dpp::DeterminantalPointProcess, z::Array{Int})
    logpmf(dpp::DeterminantalPointProcess, z::Array{Int})
    pmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    logpmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
"""


function logpmf(dpp::DeterminantalPointProcess, z::Array{Int})
    """Compute the log probability of a sample `z` under the given DPP.
    """
    L_z_eigvals = Base.LinAlg.eigvals(dpp.L[z, z])
    sum(log(L_z_eigvals)) - sum(log(dpp.Lfact.values + 1))
end


function logpmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    """Compute the log probability of a sample `z` under the given k-DPP.
    """
    L_z_eigvals = Base.LinAlg.eigvals(dpp.L[z, z])
    sum(log(L_z_eigvals)) - log(elem_symm_poly(dpp.Lfact.values, k)[end, end])
end


function pmf(dpp::DeterminantalPointProcess, z::Array{Int})
    """Compute the probability of a sample `z` under the given DPP.
    """
    exp(logpmf(dpp, z))
end


function pmf(dpp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    """Compute the probability of a sample `z` under the given DPP.
    """
    exp(logpmf(dpp, z, k))
end
