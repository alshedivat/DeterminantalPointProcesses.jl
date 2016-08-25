"""
Determinantal Point Processes.

References:
-----------
    [1] Kulesza, A., and B. Taskar. Determinantal point processes for machine
        learning. arXiv preprint arXiv:1207.6083, 2012.
    [2] Kang, B. Fast determinantal point process sampling with application to
        clustering. NIPS, 2013.
"""


function logpmf(pp::DeterminantalPointProcess, z::Array{Int})
    """Compute the log probability of a sample `z` under the given DPP.
    """
    Lsubvals = Base.LinAlg.eigvals(pp.L[z, z])
    sum(log(Lsubvals)) - sum(log(pp.Lfact.values + 1))
end


function logpmf(pp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    """Compute the log probability of a sample `z` under the given k-DPP.
    """
    Lsubvals = Base.LinAlg.eigvals(pp.L[z, z])
    sum(log(Lsubvals)) - log(elem_symm_poly(pp.Lfact.values, k)[end, end])
end


function pmf(pp::DeterminantalPointProcess, z::Array{Int})
    """Compute the probability of a sample `z` under the given DPP.
    """
    exp(logpmf(pp, z))
end


function pmf(pp::DeterminantalPointProcess, z::Array{Int}, k::Int)
    """Compute the probability of a sample `z` under the given DPP.
    """
    exp(logpmf(pp, z, k))
end


function _sample_mask(Λ::SharedArray{Float64},
                      M::SharedMatrix{Bool},
                      i::Int, seed::Int)
    """Sample a mask for an elementary DPP.
    """
    rng = MersenneTwister(seed)

    for j in 1:length(Λ)
        M[j, i] = (rand(rng) < Λ[j] / (Λ[j] + 1))
    end
end


function _sample_k_mask(Λ::SharedArray{Float64},
                        M::SharedMatrix{Bool},
                        E::SharedMatrix{Float64},
                        k::Int, i::Int, seed::Int)
    """Sample a mask for an elementary k-DPP.
    """
    rng = MersenneTwister(seed)

    j = length(Λ)
    remaining = k

    # iteratively sample a k-mask
    while remaining > 0
        # compute marginal of j given that we choose remaining values from 1:j
        if j == remaining
            marg = 1
        else
            marg = Λ[j] * E[remaining, j] / E[remaining + 1, j + 1];
        end

        # sample marginal
        if rand(rng) <= marg
            M[j, i] = true
            remaining -= 1
        end
        j -= 1
      end
end


function _sample_from_elementary(V::SharedMatrix,
                                 M::SharedMatrix{Bool},
                                 i::Int, seed::Int)
    """Exact sampling from an elementary DPP. The algorithm based on [1].
    """
    rng = MersenneTwister(seed)

    # select the elementary DPP
    V_mask = M[:, i]

    # edge case: empty sample
    if !any(V_mask)
        return Int[]
    end

    # select the kernel of the elementary DPP
    L = V[:, V_mask]

    Y = Int[]
    mask = ones(Bool, size(L, 2))
    prob = Array{Float64}(size(L, 1))

    for i in 1:size(L, 2)
        # compute probabilities
        fill!(prob, 0)
        for c in 1:size(L, 2)
            !mask[c] && continue
            for r in 1:size(L, 1)
                prob[r] += L[r, c].^2
            end
        end
        prob ./= sum(prob)

        # sample a point in the original space
        h = findfirst(rand(rng) .<= cumsum(prob))
        push!(Y, h)

        # select and mask-out an element
        j = get_first_nz_idx(L[h, :], mask)
        mask[j] = false

        if any(mask)
            # Subtract scaled Lj from other columns so that their
            # projections on e_s[i] turns into 0. This operation
            # preserves the rank of L_{-j}.
            for c in 1:size(L, 2)
                !mask[c] && continue
                for r in 1:size(L, 1)
                    L[r, c] -= L[r, j] * L[h, c] / L[h, j]
                end
            end

            # Gram-Schmidt orthogonalization
            L[:, mask] = qr(L[:, mask])[1]
        end
    end

    sort(Y)
end


function rand(pp::DeterminantalPointProcess, N::Int)
    """Exact sampling from a DPP [1].
    """
    Λ = SharedArray{Float64}(pp.Lfact.values)
    V = SharedMatrix{Float64}(pp.Lfact.vectors)
    M = SharedMatrix{Bool}(zeros(Bool, pp.size, N))

    # step I: sample masks for elementary DPPs
    pmap((i, seed) -> _sample_mask(Λ, M, i, seed),
         1:N, abs(rand(pp.rng, Int, N)))

    # step II: iteratively sample from a mixture of elementary DPPs
    pmap((i, seed) -> _sample_from_elementary(V, M, i, seed),
         1:N, abs(rand(pp.rng, Int, N)))
end


function rand(pp::DeterminantalPointProcess, N::Int, k::Int)
    """Exact sampling from a k-DPP [1].
    """
    Λ = SharedArray{Float64}(pp.Lfact.values)
    V = SharedMatrix{Float64}(pp.Lfact.vectors)
    M = SharedMatrix{Bool}(zeros(Bool, pp.size, N))

    # compute elementary symmetric polynomials
    E = SharedMatrix{Float64}(elem_symm_poly(pp.Lfact.values, k))

    # step I: sample masks for elementary DPPs
    pmap((i, seed) -> _sample_k_mask(Λ, M, E, k, i, seed),
         1:N, abs(rand(pp.rng, Int, N)))

    # step II: iteratively sample from a mixture of elementary DPPs
    pmap((i, seed) -> _sample_from_elementary(V, M, i, seed),
         1:N, abs(rand(pp.rng, Int, N)))
end


function randmcmc(pp::DeterminantalPointProcess, N::Int,
                  persistent=false,
                  mixing_time=nothing,
                  steps_between_samples=nothing,
                  eps=1e-1)
    """MCMC sampling from a DPP [2].
    """
    # TODO
end


function randmcmc(pp::DeterminantalPointProcess, N::Int, k::Int,
                  persistent=false,
                  mixing_time=nothing,
                  steps_between_samples=nothing,
                  eps=1e-1)
    """MCMC sampling from a k-DPP [2].
    """
    # TODO
end
