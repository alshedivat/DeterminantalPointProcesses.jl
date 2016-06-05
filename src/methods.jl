doc"""
Determinantal Point Processes.

References:
-----------
    [1] Kulesza, A., and B. Taskar. "Determinantal point processes for machine
        learning." arXiv preprint arXiv:1207.6083 (2012).
    [2] Kang, B. "Fast determinantal point process sampling with application to
        clustering." NIPS, 2013.
"""

function _sample_from_elementary(V::SharedMatrix, M::SharedMatrix{Bool}, i::Int)
    """Exact sampling from an elementary DPP. The algorithm based on [1].
    """
    sample = Int[]

    # select the elementary DPP
    V_mask = M[:, i]

    # edge case: empty sample
    if !any(V_mask)
        return Int[]
    end

    # select the kernel of the elementary DPP
    L = V[:, V_mask]
    mask = ones(Bool, size(L, 2))

    prob = zeros(size(L, 1))
    for j in 1:size(L, 2)
        # compute probabilities
        fill!(prob, 0)
        for c in 1:size(L, 2)
            !mask[c] && continue
            for r in 1:size(L, 1)
                prob[r] += L[r, c].^2
            end
        end
        prob ./= size(L, 2) - j + 1

        # for dbg purposes
        # @assert abs(sum(prob) - 1) < size(L, 1) * eps()

        # sample a point in the original space
        point = rand(Multinomial(1, prob))
        h = get_first_nz_idx(point)
        push!(sample, h)

        # select and mask-out an element
        mask[get_first_nz_idx(L[h, :], mask)] = false

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
            L[:, mask], _ = qr(L[:, mask])
        end
    end

    sample
end


function rand(pp::DeterminantalPointProcess, N::Int)
    """Exact sampling from DPP. The standard algorithm based on [1].
    """
    V = SharedMatrix{Float64}(pp.kernel.vectors)
    M = SharedMatrix{Bool}(zeros(Bool, pp.size, N))

    # step I: sample masks for elementary DPPs
    for n in 1:N
        for j in 1:pp.size
            λ = pp.kernel.values[j]
            M[j, n] = (rand(pp.rng) < λ / (λ + 1))
        end
    end

    # step II: iteratively sample from a mixture of elementary DPPs
    pmap((i) -> _sample_from_elementary(V, M, i), 1:N)
end


# function sample(pp::DeterminantalPointProcess, N::Int, k::Int)
#     """Exact sampling from k-DPP. The standard algorithm based on [1].
#     """
#     # TODO
# end


# function sample(pp::DeterminantalPointProcess, N::Int;
#                 persistent=false,
#                 mixing_time=nothing,
#                 steps_between_samples=nothing,
#                 eps=1e-1)
#     """Fast sampling from DPP. The algorithm is based on [2].
#     """
#     # TODO
# end


# function sample(pp::DeterminantalPointProcess, N::Int, k::Int;
#                 persistent=false,
#                 mixing_time=nothing,
#                 steps_between_samples=nothing,
#                 eps=1e-1)
#     """Fast sampling from k-DPP. The algorithm based on [2].
#     """
#     # TODO
# end
