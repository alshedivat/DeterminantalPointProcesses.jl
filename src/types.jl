# abstract types
abstract PointProcess


# specific types
type DeterminantalPointProcess <: PointProcess
    L::Symmetric
    Lfact::Eigen
    size::Int
    rng::AbstractRNG

    function DeterminantalPointProcess(L::Symmetric; seed::Int=42)
        Lfact = Base.LinAlg.eigfact(L)
        new(L, Lfact, length(Lfact.values), MersenneTwister(seed))
    end

    function DeterminantalPointProcess(Lfact::Eigen; seed::Int=42)
        L = Symmetric((Lfact.vectors .* Lfact.values') * Lfact.vectors')
        new(L, Lfact, length(Lfact.values), MersenneTwister(seed))
    end
end


type KroneckerDeterminantalPointProcess <: PointProcess
    # TODO
end


# aliases
typealias DPP DeterminantalPointProcess
typealias KDPP KroneckerDeterminantalPointProcess
typealias MCMCState Tuple{BitArray{1}, Array{Float64, 2}}
