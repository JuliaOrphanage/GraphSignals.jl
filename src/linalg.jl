function adjacency_matrix(adj::AbstractMatrix{T}, ::Type{S}) where {T,S}
    _dim_check(adj)
    return Matrix{S}(adj)
end

function adjacency_matrix(adj::AbstractMatrix)
    _dim_check(adj)
    return Array(adj)
end

adjacency_matrix(adj::Matrix{T}, ::Type{T}) where {T} = adjacency_matrix(adj)

function adjacency_matrix(adj::Matrix)
    _dim_check(adj)
    return adj
end

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

function _dim_check(adj)
    m, n = size(adj)
    (m == n) || throw(DimensionMismatch("adjacency matrix is not a square matrix: ($m, $n)"))
end


"""
    degrees(g, [T]; dir=:out)

Degree of each vertex. Return a vector which contains the degree of each vertex in graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `SimpleGraph`, `SimpleDiGraph` (from Graphs) or
    `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).

# Examples

```jldoctest
julia> using GraphSignals

julia> m = [0 1 1; 1 0 0; 1 0 0];

julia> GraphSignals.degrees(m)
3-element Vector{Int64}:
 2
 1
 1
```
"""
function degrees(g, ::Type{T}=eltype(g); dir::Symbol=:out) where {T}
    adj = adjacency_matrix(g, T)
    if issymmetric(adj)
        d = vec(sum(adj, dims=1))
    else
        if dir == :out
            d = vec(sum(adj, dims=1))
        elseif dir == :in
            d = vec(sum(adj, dims=2))
        elseif dir == :both
            d = vec(sum(adj, dims=1)) + vec(sum(adj, dims=2))
        else
            throw(ArgumentError("dir only accept :in, :out or :both, but got $(dir)."))
        end
    end
    return T.(d)
end

degrees(adj::CuSparseMatrixCSC, ::Type{T}=eltype(adj); dir::Symbol=:out) where {T} =
    degrees(CuMatrix{T}(adj); dir=dir)

"""
    degree_matrix(g, [T]; dir=:out)

Degree matrix of graph `g`. Return a matrix which contains degrees of each vertex in its diagonal.
The values other than diagonal are zeros.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).

# Examples

```jldoctest
julia> using GraphSignals

julia> m = [0 1 1; 1 0 0; 1 0 0];

julia> GraphSignals.degree_matrix(m)
3×3 LinearAlgebra.Diagonal{Int64, Vector{Int64}}:
 2  ⋅  ⋅
 ⋅  1  ⋅
 ⋅  ⋅  1
```
"""
function degree_matrix(g, ::Type{T}=eltype(g);
                       dir::Symbol=:out, squared::Bool=false, inverse::Bool=false) where {T}
    d = degrees(g, T, dir=dir)
    squared && (d .= sqrt.(d))
    inverse && (d .= safe_inv.(d))
    return Diagonal(T.(d))
end

safe_inv(x::T) where {T} = ifelse(iszero(x), zero(T), inv(x))

"""
    normalized_adjacency_matrix(g, [T]; selfloop=false)

Normalized adjacency matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `selfloop`: adding self loop while calculating the matrix (optional).
"""
function normalized_adjacency_matrix(g, ::Type{T}=eltype(g); selfloop::Bool=false) where {T}
    adj = adjacency_matrix(g, T)
    selfloop && (adj += I)
    inv_sqrtD = degree_matrix(g, T, dir=:both, squared=true, inverse=true)
    return inv_sqrtD * adj * inv_sqrtD
end

"""
    laplacian_matrix(g, [T]; dir=:out)

Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
Graphs.laplacian_matrix(g, ::Type{T}=eltype(g); dir::Symbol=:out) where {T} =
    degree_matrix(g, T, dir=dir) - adjacency_matrix(g, T)

"""
    normalized_laplacian(g, [T]; dir=:both, selfloop=false)

Normalized Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `selfloop`: adding self loop while calculating the matrix (optional).
- `dir`: direction of graph; should be `:in` or `:out` (optional).
"""
function normalized_laplacian(g, ::Type{T}=float(eltype(g));
                              dir::Symbol=:both, selfloop::Bool=false) where {T}
    L = adjacency_matrix(g, T)
    if dir == :both
        selfloop && (L += I)
        inv_sqrtD = degree_matrix(g, T, dir=:both, squared=true, inverse=true)
        L .= I - inv_sqrtD * L * inv_sqrtD
    else
        inv_D = degree_matrix(g, T, dir=dir, inverse=true)
        L .= I - inv_D * L
    end
    return L
end

@doc raw"""
    scaled_laplacian(g, [T])

Scaled Laplacien matrix of graph `g`,
defined as ``\hat{L} = \frac{2}{\lambda_{max}} L - I`` where ``L`` is the normalized Laplacian matrix.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
"""
function scaled_laplacian(g, ::Type{T}=float(eltype(g))) where {T}
    adj = adjacency_matrix(g, T)
    # @assert issymmetric(adj) "scaled_laplacian only works with symmetric matrices"
    E = eigen(Symmetric(Array(adj))).values
    return T(2. / maximum(E)) .* normalized_laplacian(adj, T) - I
end

"""
    random_walk_laplacian(g, [T]; dir=:out)

Random walk normalized Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
function random_walk_laplacian(g, ::Type{T}=float(eltype(g)); dir::Symbol=:out) where {T}
    inv_D = degree_matrix(g, T; dir=dir, inverse=true)
    A = adjacency_matrix(g, T)
    P = inv_D * A
    return SparseMatrixCSC(I - P)
end

"""
    signless_laplacian(g, [T]; dir=:out)

Signless Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
signless_laplacian(g, ::Type{T}=eltype(g); dir::Symbol=:out) where {T} =
    degree_matrix(g, T, dir=dir) + adjacency_matrix(g, T)
