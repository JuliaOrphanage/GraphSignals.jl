random_walk(sg::GraphSignals.SparseGraph{B,T}, start::Int, n::Int=1) where {B,T<:CuSparseMatrixCSC} =
    random_walk(collect(sg.S), start, n)

neighbor_sample(sg::GraphSignals.SparseGraph{B,T}, start::Int, n::Int=1; replace::Bool=false) where {B,T<:CuSparseMatrixCSC} =
    neighbor_sample(collect(sg.S), start, n; replace=replace)
