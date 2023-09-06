sparsecsc(A::AbstractMatrix) = sparse(A)

SparseArrays.getcolptr(S::SparseMatrixCSC, col::Integer) = S.colptr[col]:(S.colptr[col+1]-1)
SparseArrays.getcolptr(S::SparseMatrixCSC, I::UnitRange) = S.colptr[I.start]:(S.colptr[I.stop+1]-1)

SparseArrays.rowvals(S::SparseMatrixCSC, col::Integer) = _rowvals(S, col)
SparseArrays.rowvals(S::SparseMatrixCSC, I::UnitRange) = _rowvals(S, I)
rowvalview(S::SparseMatrixCSC, col::Integer) = _rowvalview(S, col)
rowvalview(S::SparseMatrixCSC, I::UnitRange) = _rowvalview(S, I)

_rowvals(S, col) = rowvals(S)[SparseArrays.getcolptr(S, col)]
_rowvalview(S, col) = view(rowvals(S), SparseArrays.getcolptr(S, col))

SparseArrays.nonzeros(S::SparseMatrixCSC, col::Integer) = _nonzeros(S, col)
SparseArrays.nonzeros(S::SparseMatrixCSC, I::UnitRange) = _nonzeros(S, I)
SparseArrays.nzvalview(S::SparseMatrixCSC, col::Integer) = _nzvalview(S, col)
SparseArrays.nzvalview(S::SparseMatrixCSC, I::UnitRange) = _nzvalview(S, I)

_nonzeros(S, col) = nonzeros(S)[SparseArrays.getcolptr(S, col)]
_nzvalview(S, col) = view(nonzeros(S), SparseArrays.getcolptr(S, col))

"""
    colvals(S, [n]; upper_traingle=false)

Returns column indices of nonzero values in a sparse array `S`.
Nonzero values are count up to column `n`. If `n` is not specified,
all nonzero values are considered.

# Arguments

- `S::SparseCSC`: Sparse array, which can be `SparseMatrixCSC` or `CuSparseMatrixCSC`.
- `n::Int`: Maximum columns to count nonzero values.
- `upper_traingle::Bool`: To count nonzero values in upper traingle only or not.
"""
colvals(S::SparseMatrixCSC; upper_traingle::Bool=false) =
    colvals(S, size(S, 2); upper_traingle=upper_traingle)

colvals(S::SparseMatrixCSC, n::Int; upper_traingle::Bool=false) =
    _colvals(S, n; upper_traingle=upper_traingle)

function _colvals(S, n::Int; upper_traingle::Bool=false)
    if upper_traingle
        ls = [count(rowvalview(S, j) .≤ j) for j in 1:n]
        pushfirst!(ls, 1)
        cumsum!(ls, ls)
        l = ls[end]-1
    else
        colptr = collect(SparseArrays.getcolptr(S))
        ls = view(colptr, 2:(n+1)) - view(colptr, 1:n)
        pushfirst!(ls, 1)
        cumsum!(ls, ls)
        l = length(rowvals(S))
    end
    return _fill_colvals(rowvals(S), ls, l, n)
end

function _fill_colvals(tmpl::AbstractVector, ls, l::Int, n::Int)
    res = similar(tmpl, l)
    for j in 1:n
        fill!(view(res, ls[j]:(ls[j+1]-1)), j)
    end
    return res
end

"""
Transform a regular cartesian index `A[i, j]` into a CSC-compatible index `spA.nzval[idx]`.
"""
get_csc_index(S::SparseMatrixCSC, i::Integer, j::Integer) = _get_csc_index(S, i, j)

function _get_csc_index(S, i::Integer, j::Integer)
    idx1 = SparseArrays.getcolptr(S, j)
    row = view(rowvals(S), idx1)
    idx2 = findfirst(x -> x == i, row)
    return idx1[idx2]
end

"""
Order the edges in a graph by giving a unique integer to each edge.
"""
order_edges(S::SparseMatrixCSC; directed::Bool=false) = _order_edges(S; directed=directed)

_order_edges(S::SparseMatrixCSC; directed::Bool=false) = _order_edges!(similar(rowvals(S)), S, Val(directed))

order_edges!(edges, S::SparseMatrixCSC, ::Val{false}) = _order_edges!(edges, S, Val(false))
order_edges!(edges, S, ::Val{true}) = _order_edges!(edges, S, Val(true))

function _order_edges!(edges, S, ::Val{false})
    @assert issymmetric(S) "Matrix of undirected graph must be symmetric."
    k = 1
    for j in axes(S, 2)
        idx1 = SparseArrays.getcolptr(S, j)
        row = rowvalview(S, j)
        for idx2 in 1:length(row)
            idx = idx1[idx2]
            i = row[idx2]
            if i < j  # upper triangle
                edges[idx] = k
                edges[get_csc_index(S, j, i)] = k
                k += 1
            elseif i == j  # diagonal
                edges[idx] = k
                k += 1
            end
        end
    end
    return edges
end

function _order_edges!(edges::T, S, ::Val{true}) where {T}
    edges .= T(1:length(edges))
    edges
end
