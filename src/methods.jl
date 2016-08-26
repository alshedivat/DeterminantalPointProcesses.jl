"""
Determinantal Point Processes.

References:
-----------
    [1] Kulesza, A., and B. Taskar. Determinantal point processes for machine
        learning. arXiv preprint arXiv:1207.6083, 2012.
    [2] Kang, B. Fast determinantal point process sampling with application to
        clustering. NIPS, 2013.
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


function rand(dpp::DeterminantalPointProcess, N::Int)
    """Exact sampling from a DPP [1].
    """
    Λ = SharedArray{Float64}(dpp.Lfact.values)
    V = SharedMatrix{Float64}(dpp.Lfact.vectors)
    M = SharedMatrix{Bool}(zeros(Bool, dpp.size, N))

    # step I: sample masks for elementary DPPs
    pmap((i, seed) -> _sample_mask(Λ, M, i, seed),
         1:N, abs(rand(dpp.rng, Int, N)))

    # step II: iteratively sample from a mixture of elementary DPPs
    pmap((i, seed) -> _sample_from_elementary(V, M, i, seed),
         1:N, abs(rand(dpp.rng, Int, N)))
end


function rand(dpp::DeterminantalPointProcess, N::Int, k::Int)
    """Exact sampling from a k-DPP [1].
    """
    Λ = SharedArray{Float64}(dpp.Lfact.values)
    V = SharedMatrix{Float64}(dpp.Lfact.vectors)
    M = SharedMatrix{Bool}(zeros(Bool, dpp.size, N))

    # compute elementary symmetric polynomials
    E = SharedMatrix{Float64}(elem_symm_poly(dpp.Lfact.values, k))

    # step I: sample masks for elementary DPPs
    pmap((i, seed) -> _sample_k_mask(Λ, M, E, k, i, seed),
         1:N, abs(rand(dpp.rng, Int, N)))

    # step II: iteratively sample from a mixture of elementary DPPs
    pmap((i, seed) -> _sample_from_elementary(V, M, i, seed),
         1:N, abs(rand(dpp.rng, Int, N)))
end


function _do_mcmc_step!(dpp::DeterminantalPointProcess, state::MCMCState)
    """Perform one MCMC accept-reject transition.
    """
    # propose an element to swap
    u = rand(dpp.rng, 1:dpp.size)
    insert = !state[1][u]

    # attempt to make a transition
    if insert
        p = _comp_accept_prob(dpp, state, u, insert)
        rand(dpp.rng) < p && _update_mcmc_state!(dpp, state, u, insert)
    else  # delete
        new_state = _update_mcmc_state(dpp, state, u, insert)
        p = _comp_accept_prob(dpp, new_state, u, insert)
        if rand(dpp.rng) < p
            copy!(state[1], new_state[1])
            copy!(state[2], new_state[2])
        end
    end
end


function _comp_accept_prob(dpp::DeterminantalPointProcess, state::MCMCState,
                           u::Int, insert::Bool)
    """Compute accept probability to insert / delete u from the state.
    """
    z, L_z_inv = state

    d_u = dpp.L[u, u]
    if any(z)
        b_u = dpp.L[z, u]
        d_u -= dot(b_u, L_z_inv[z, z] * b_u)
    end

    insert ? min(1.0, d_u) : min(1.0, 1.0 / d_u)
end


function _update_mcmc_state!(dpp::DeterminantalPointProcess, state::MCMCState,
                             u::Int, insert::Bool)
    """Compute Sherman-Morrison-Woodbury update for L_z_inv after transition.
    """
    z, L_z_inv = state

    if insert
        d_u = dpp.L[u, u]
        if any(z)
            b_u = dpp.L[z, u]
            x_u = L_z_inv[z, z] * b_u
            d_u -= dot(b_u, x_u)

            L_z_inv[z, z] += (x_u * x_u') / d_u
            L_z_inv[z, u] = L_z_inv[u, z] = - x_u / d_u
        end

        L_z_inv[u, u] = 1. / d_u
        z[u] = true
    else  # delete
        z[u] = false

        e = L_z_inv[z, u]
        f = L_z_inv[u, u]

        L_z_inv[z, z] -= (e * e') / f
    end
end


function _update_mcmc_state(dpp::DeterminantalPointProcess, state::MCMCState,
                            u::Int, insert::Bool)
    """Compute Sherman-Morrison-Woodbury update for L_z_inv after transition.
    """
    new_state = deepcopy(state)
    _update_mcmc_state!(dpp, new_state, u, insert)
    new_state
end


function randmcmc(dpp::DeterminantalPointProcess, N::Int;
                  init_state=nothing,
                  return_final_state::Bool=false,
                  mixing_steps::Int=ceil(Int, dpp.size*log(dpp.size/mix_eps)),
                  steps_between_samples::Int=mixing_steps,
                  mix_eps::Float64=1e-1)
    """MCMC sampling from a DPP [2].

    TODO: Add support for running MCMC in parallel, similar as rand.
          Make sure parallelization produces unbiased and consistent samples.
    """
    # initialize the Markov chain
    state = init_state
    if state == nothing
        L_z_inv = Array{Float64}(size(dpp.L))
        z = bitrand(dpp.rng, dpp.size)  # TODO: improve initialization (?)
        if any(z)
            L_z_inv[z, z] = pinv(dpp.L[z, z])
        end
        state = (z, L_z_inv)
    end

    # sanity check
    @assert typeof(state) == MCMCState

    # mix the Markov chain
    for t in 1:mixing_steps
        _do_mcmc_step!(dpp, state)
    end

    Y = []
    for i in 1:N
        push!(Y, find(state[1]))
        for t in 1:steps_between_samples
            _do_mcmc_step!(dpp, state)
        end
    end

    return_final_state ? (Y, state) : Y
end


function randmcmc(dpp::DeterminantalPointProcess, N::Int, k::Int,
                  init_state=nothing,
                  return_final_state::Bool=false,
                  mixing_steps::Int=ceil(Int, dpp.size*log(dpp.size/mix_eps)),
                  steps_between_samples::Int=mixing_steps,
                  mix_eps::Float64=1e-1)
    """MCMC sampling from a k-DPP [2].
    """
    # TODO
end
