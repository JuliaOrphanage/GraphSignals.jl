function SparseGraph(A::CuSparseMatrixCSC{Tv}, edges::AnyCuVector{Ti}, directed::Bool) where {Tv,Ti}
    E = maximum(edges)
    return SparseGraph{directed,typeof(A),typeof(edges),typeof(E)}(A, edges, E)
end

SparseGraph(A::CuSparseMatrixCSC, directed::Bool, ::Type{T}=eltype(A)) where {T} =
    SparseGraph(A, order_edges(A, directed=directed), directed, T)
