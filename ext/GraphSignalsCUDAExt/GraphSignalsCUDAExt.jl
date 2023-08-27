module GraphSignalsCUDAExt

using SparseArrays

using CUDA, CUDA.CUSPARSE

using GraphSignals

include("linalg.jl")
include("sparsematrix.jl")
include("sparsegraph.jl")
include("random.jl")

end
