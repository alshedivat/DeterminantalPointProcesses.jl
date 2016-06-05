# abstrac types
abstract PointProcess

# specific types
type DeterminantalPointProcess <: PointProcess
    kernel::Eigen
    size::Int
    rng::AbstractRNG

    function DeterminantalPointProcess(L::Symmetric, seed::Int = 42)
        kernel = Base.LinAlg.eigfact(L)
        new(kernel, length(kernel.values), MersenneTwister(seed))
    end

    function DeterminantalPointProcess(kernel::Eigen, seed::Int = 42)
        new(kernel, length(kernel.values), MersenneTwister(seed))
    end
end


type KroneckerDeterminantalPointProcess <: PointProcess
    # TODO
end

# aliases
# typealias DPP DeterminantalPointProcess
# typealias KDPP KroneckerDeterminantalPointProcess
