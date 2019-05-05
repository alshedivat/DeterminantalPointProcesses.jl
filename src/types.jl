# abstract types
abstract type PointProcess end;


# specific types
mutable struct DeterminantalPointProcess <: PointProcess
    L::Symmetric
    Lfact::Eigen
    size::Int
    rng::AbstractRNG

    function DeterminantalPointProcess(L::Symmetric; seed::Int=42)
        Lfact = eigen(L)
        new(L, Lfact, length(Lfact.values), MersenneTwister(seed))
    end

    function DeterminantalPointProcess(Lfact::Eigen; seed::Int=42)
        L = Symmetric((Lfact.vectors .* Lfact.values') * Lfact.vectors')
        new(L, Lfact, length(Lfact.values), MersenneTwister(seed))
    end
end


mutable struct KroneckerDeterminantalPointProcess <: PointProcess
    # TODO
end


# aliases
const DPP = DeterminantalPointProcess
const KDPP = KroneckerDeterminantalPointProcess
const MCMCState = Tuple{BitArray{1}, Array{Float64, 2}}
