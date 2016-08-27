using DeterminantalPointProcesses
using FactCheck

import Iterators: subsets
import Combinatorics: combinations


rng = MersenneTwister(42)

n = 5
k = 2
A = rand(rng, n, n)
L = Symmetric(A * A')
dpp = DPP(L)


facts("Ensure correct pmf and logpmf computation") do
    context("For DPP") do
        # compute the true distribution
        true_pmf = Float64[]
        true_logpmf = Float64[]
        for z in subsets(1:n)
            push!(true_pmf, pmf(dpp, z))
            push!(true_logpmf, logpmf(dpp, z))
        end
        @fact sum(true_pmf) --> roughly(1.0)
        @fact all(true_pmf .<= 1.0) --> true
        @fact all(true_logpmf .<= 0.0) --> true
    end

    context("For k-DPP") do
        # compute the true distribution
        true_pmf = Float64[]
        true_logpmf = Float64[]
        for z in combinations(1:n, k)
            push!(true_pmf, pmf(dpp, z, k))
            push!(true_logpmf, logpmf(dpp, z, k))
        end
        @fact sum(true_pmf) --> roughly(1.0)
        @fact all(true_pmf .<= 1.0) --> true
        @fact all(true_logpmf .<= 0.0) --> true
    end
end


facts("Ensure correct sampling from DPP") do
    # compute the true distribution
    all_subsets = []
    true_pmf = Float64[]
    for (i, z) in enumerate(subsets(1:n))
        push!(true_pmf, pmf(dpp, z))
        push!(all_subsets, (z, i))
    end
    subset_to_idx = Dict(all_subsets)

    context("Exact sampling") do
        nb_samples = 1000
        samples = rand(dpp, nb_samples)

        # compute the empirical distribution
        empirical_pmf = zeros(Float64, length(true_pmf))
        for z in samples
            empirical_pmf[subset_to_idx[z]] += 1
        end
        empirical_pmf ./= nb_samples

        # ensure that the empirical pmf is close to the true pmf
        total_variation = maximum(abs(true_pmf - empirical_pmf))
        @fact total_variation --> roughly(0.0; atol=1e-1)
    end

    context("MCMC sampling") do
        nb_samples = 1000
        samples, state = randmcmc(dpp, nb_samples, return_final_state=true)

        # ensure that L_z_inv makes sense (i.e., noise did not accumulate)
        z, L_z_inv = state
        @fact L_z_inv[z, z] * dpp.L[z, z] --> roughly(eye(sum(z)))

        # compute the empirical distribution
        empirical_pmf = zeros(Float64, length(true_pmf))
        for z in samples
            empirical_pmf[subset_to_idx[z]] += 1
        end
        empirical_pmf ./= nb_samples

        # ensure that the empirical pmf is close to the true pmf
        total_variation = maximum(abs(true_pmf - empirical_pmf))
        @fact total_variation --> roughly(0.0; atol=1e-1)
    end
end


facts("Ensure correct sampling from k-DPP") do
    # compute the true distribution
    all_k_subsets = []
    true_pmf = Float64[]
    for (i, z) in enumerate(combinations(1:n, k))
        push!(true_pmf, pmf(dpp, z))
        push!(all_k_subsets, (z, i))
    end
    true_pmf ./= sum(true_pmf)
    k_subset_to_idx = Dict(all_k_subsets)

    context("Exact sampling") do
        nb_samples = 10000
        samples = rand(dpp, nb_samples, k)

        # ensure that samples are of proper cardinality
        @fact all(map(length, samples) .== k) --> true

        # compute the empirical distribution
        empirical_pmf = zeros(Float64, length(true_pmf))
        for z in samples
            empirical_pmf[k_subset_to_idx[z]] += 1
        end
        empirical_pmf ./= nb_samples

        # ensure that the empirical pmf is close to the true pmf
        total_variation = maximum(abs(true_pmf - empirical_pmf))
        @fact total_variation --> roughly(0.0; atol=1e-1)
    end

    context("MCMC sampling") do
        nb_samples = 1000
        samples, state = randmcmc(dpp, nb_samples, k, return_final_state=true)

        # ensure that L_z_inv makes sense (i.e., noise did not accumulate)
        z, L_z_inv = state
        @fact L_z_inv[z, z] * dpp.L[z, z] --> roughly(eye(sum(z)))

        # ensure that samples are of proper cardinality
        @fact all(map(length, samples) .== k) --> true

        # compute the empirical distribution
        empirical_pmf = zeros(Float64, length(true_pmf))
        for z in samples
            empirical_pmf[k_subset_to_idx[z]] += 1
        end
        empirical_pmf ./= nb_samples

        # ensure that the empirical pmf is close to the true pmf
        total_variation = maximum(abs(true_pmf - empirical_pmf))
        @fact total_variation --> roughly(0.0; atol=1e-1)
    end
end
