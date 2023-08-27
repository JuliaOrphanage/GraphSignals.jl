function adjacency_matrix(adj::CuSparseMatrixCSC{T}, ::Type{S}) where {T,S}
    _dim_check(adj)
    return CuMatrix{S}(collect(adj))
end

function adjacency_matrix(adj::CuSparseMatrixCSC)
    _dim_check(adj)
    return CuMatrix(adj)
end

adjacency_matrix(adj::CuMatrix{T}, ::Type{T}) where {T} = adjacency_matrix(adj)

function adjacency_matrix(adj::CuMatrix)
    _dim_check(adj)
    return adj
end

degrees(adj::CuSparseMatrixCSC, ::Type{T}=eltype(adj); dir::Symbol=:out) where {T} =
    degrees(CuMatrix{T}(adj); dir=dir)
