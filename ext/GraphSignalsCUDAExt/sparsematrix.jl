sparsecsc(A::AnyCuMatrix) = CuSparseMatrixCSC(A)

SparseArrays.rowvals(S::CuSparseMatrixCSC, col::Integer) = rowvals(S)[SparseArrays.getcolptr(S, col)]
SparseArrays.rowvals(S::CuSparseMatrixCSC, I::UnitRange) = rowvals(S)[SparseArrays.getcolptr(S, I)]
rowvalview(S::CuSparseMatrixCSC, col::Integer) = view(rowvals(S), SparseArrays.getcolptr(S, col))
rowvalview(S::CuSparseMatrixCSC, I::UnitRange) = view(rowvals(S), SparseArrays.getcolptr(S, I))

# TODO: @allowscalar should be removed.
SparseArrays.getcolptr(S::CuSparseMatrixCSC) = S.colPtr
SparseArrays.getcolptr(S::CuSparseMatrixCSC, col::Integer) = CUDA.@allowscalar S.colPtr[col]:(S.colPtr[col+1]-1)

SparseArrays.nonzeros(S::CuSparseMatrixCSC, col::Integer) = _nonzeros(S, col)
SparseArrays.nonzeros(S::CuSparseMatrixCSC, I::UnitRange) = _nonzeros(S, I)
SparseArrays.nzvalview(S::CuSparseMatrixCSC, col::Integer) = _nzvalview(S, col)
SparseArrays.nzvalview(S::CuSparseMatrixCSC, I::UnitRange) = _nzvalview(S, I)

colvals(S::CuSparseMatrixCSC; upper_traingle::Bool=false) =
    colvals(S, size(S, 2); upper_traingle=upper_traingle)

colvals(S::CuSparseMatrixCSC, n::Int; upper_traingle::Bool=false) =
    _colvals(S, n; upper_traingle=upper_traingle)

get_csc_index(S::CuSparseMatrixCSC, i::Integer, j::Integer) = _get_csc_index(S, i, j)

order_edges(S::CuSparseMatrixCSC; directed::Bool=false) = _order_edges(S; directed=directed)

order_edges!(edges, S::CuSparseMatrixCSC, ::Val{false}) = _order_edges!(edges, S, Val(false))