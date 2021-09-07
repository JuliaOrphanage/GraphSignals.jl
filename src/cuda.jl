SparseArrays.getcolptr(S::CuSparseMatrixCSC, col::Integer) = CUDA.@allowscalar S.colPtr[col]:(S.colPtr[col+1]-1)
SparseArrays.getcolptr(S::CuSparseMatrixCSC, I::UnitRange) = CUDA.@allowscalar S.colPtr[I.start]:(S.colPtr[I.stop+1]-1)

promote_graph(graph::AbstractMatrix, nf::AnyCuArray) = cu(graph)

function SparseGraph(A::CuSparseMatrixCSC{Tv}, edges::AnyCuVector{Ti}, directed::Bool) where {Tv,Ti}
    E = maximum(edges)
    return SparseGraph{directed,typeof(A),typeof(edges),typeof(E)}(A, edges, E)
end

SparseGraph(A::CuSparseMatrixCSC, directed::Bool) = SparseGraph(A, order_edges(A, directed=directed), directed)
SparseGraph(A::AnyCuMatrix, directed::Bool) = SparseGraph(CuSparseMatrixCSC(A), directed)

function order_edges!(edges, S::CuSparseMatrixCSC, directed::Val{true})
    edges .= CuVector(1:length(edges))
    edges
end
