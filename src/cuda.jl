# TODO: @allowscalar should be removed.
SparseArrays.getcolptr(S::CuSparseMatrixCSC, col::Integer) = CUDA.@allowscalar S.colPtr[col]:(S.colPtr[col+1]-1)
SparseArrays.getcolptr(S::CuSparseMatrixCSC, I::UnitRange) = CUDA.@allowscalar S.colPtr[I.start]:(S.colPtr[I.stop+1]-1)

function SparseGraph(A::CuSparseMatrixCSC{Tv}, edges::AnyCuVector{Ti}, directed::Bool) where {Tv,Ti}
    E = maximum(edges)
    return SparseGraph{directed,typeof(A),typeof(edges),typeof(E)}(A, edges, E)
end
