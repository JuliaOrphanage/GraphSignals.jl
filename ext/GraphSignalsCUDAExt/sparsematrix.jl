sparsecsc(A::AnyCuMatrix) = CuSparseMatrixCSC(A)

SparseArrays.rowvals(S::CuSparseMatrixCSC, col::Integer) = rowvals(S)[SparseArrays.getcolptr(S, col)]
SparseArrays.rowvals(S::CuSparseMatrixCSC, I::UnitRange) = rowvals(S)[SparseArrays.getcolptr(S, I)]
GraphSignals.rowvalview(S::CuSparseMatrixCSC, col::Integer) = view(rowvals(S), SparseArrays.getcolptr(S, col))
GraphSignals.rowvalview(S::CuSparseMatrixCSC, I::UnitRange) = view(rowvals(S), SparseArrays.getcolptr(S, I))

# TODO: @allowscalar should be removed.
SparseArrays.getcolptr(S::CuSparseMatrixCSC) = S.colPtr
SparseArrays.getcolptr(S::CuSparseMatrixCSC, col::Integer) = CUDA.@allowscalar S.colPtr[col]:(S.colPtr[col+1]-1)

SparseArrays.nonzeros(S::CuSparseMatrixCSC, col::Integer) = GraphSignals._nonzeros(S, col)
SparseArrays.nonzeros(S::CuSparseMatrixCSC, I::UnitRange) = GraphSignals._nonzeros(S, I)
SparseArrays.nzvalview(S::CuSparseMatrixCSC, col::Integer) = _nzvalview(S, col)
SparseArrays.nzvalview(S::CuSparseMatrixCSC, I::UnitRange) = _nzvalview(S, I)

GraphSignals.colvals(S::CuSparseMatrixCSC; upper_traingle::Bool=false) =
    GraphSignals.colvals(S, size(S, 2); upper_traingle=upper_traingle)

GraphSignals.colvals(S::CuSparseMatrixCSC, n::Int; upper_traingle::Bool=false) =
    GraphSignals._colvals(S, n; upper_traingle=upper_traingle)

GraphSignals.get_csc_index(S::CuSparseMatrixCSC, i::Integer, j::Integer) = _get_csc_index(S, i, j)

GraphSignals.order_edges(S::CuSparseMatrixCSC; directed::Bool=false) = _order_edges(S; directed=directed)

GraphSignals.order_edges!(edges, S::CuSparseMatrixCSC, ::Val{false}) = _order_edges!(edges, S, Val(false))
