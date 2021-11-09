function adjacency_matrix(adj::AbstractMatrix)
    m, n = size(adj)
    (m == n) || throw(DimensionMismatch("adjacency matrix is not a square matrix: ($m, $n)"))
    return collect(adj)
end

function adjacency_matrix(adj::AnyCuMatrix)
    m, n = size(adj)
    (m == n) || throw(DimensionMismatch("adjacency matrix is not a square matrix: ($m, $n)"))
    return CuArray(adj)
end


"""
    degrees(g[, T]; dir=:out)

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

julia> GraphLaplacians.degrees(m)
3-element Vector{Int64}:
 2
 1
 1
```
"""
function degrees(adj::AbstractMatrix; dir::Symbol=:out)
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
            throw(DomainError(dir, "invalid argument, only accept :in, :out and :both"))
        end
    end
    d
end

degrees(adj::AbstractMatrix, T::DataType; dir::Symbol=:out) = degrees(T.(adj); dir=dir)

function degrees(g::AbstractGraph, T::DataType=eltype(g); dir::Symbol=:out)
    adj = Graphs.adjacency_matrix(g, T; dir=dir)
    degrees(adj, T; dir=dir)
end

"""
    degree_matrix(g[, T]; dir=:out)

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
function degree_matrix(adj::AbstractMatrix, T::DataType=eltype(adj);
                       dir::Symbol=:out, squared::Bool=false, inverse::Bool=false)
    d = degrees(adj, T, dir=dir)
    squared && (d .= sqrt.(d))
    inverse && (d .= inv.(d); replace!(d, typemax(T)=>zero(T)))
    return Diagonal(T.(d))
end

function degree_matrix(g::AbstractGraph, T::DataType=eltype(g); dir::Symbol=:out)
    adj = Graphs.adjacency_matrix(g, T; dir=dir)
    degree_matrix(adj, T; dir=dir)
end

"""
    normalized_adjacency_matrix(g[, T]; selfloop=false)

Normalized adjacency matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `selfloop`: adding self loop while calculating the matrix (optional).
"""
function normalized_adjacency_matrix(adj::AbstractMatrix, T::DataType=eltype(adj);
                                     selfloop::Bool=false)
    selfloop && (adj += I)
    inv_sqrtD = degree_matrix(adj, T, dir=:both, squared=true, inverse=true)
    return inv_sqrtD * Graphs.adjacency_matrix(adj, T) * inv_sqrtD
end

function normalized_adjacency_matrix(g::AbstractGraph, T::DataType=eltype(adj);
                                     selfloop::Bool=false)
    adj = Graphs.adjacency_matrix(g, T; dir=:both)
    return normalized_adjacency_matrix(adj, T; selfloop=selfloop)
end

"""
    laplacian_matrix(g[, T]; dir=:out)

Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
Graphs.laplacian_matrix(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out) =
    degree_matrix(adj, T, dir=dir) - SparseMatrixCSC(T.(adj))

"""
    normalized_laplacian(g[, T]; dir=:both, selfloop=false)

Normalized Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `selfloop`: adding self loop while calculating the matrix (optional).
- `dir`: direction of graph; should be `:in` or `:out` (optional).
"""
function normalized_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj);
                              dir::Symbol=:both, selfloop::Bool=false)
    if dir == :both
        selfloop && (adj += I)
        inv_sqrtD = degree_matrix(adj, T, dir=:both, squared=true, inverse=true)
        return T.(I - inv_sqrtD * adj * inv_sqrtD)
    else
        return T.(I - degree_matrix(adj, T, dir=dir, inverse=true) * adj)
    end
end

function normalized_laplacian(g::AbstractGraph, T::DataType=eltype(g);
                              dir::Symbol=:both, selfloop::Bool=false)
    adj = Graphs.adjacency_matrix(g, T)
    return normalized_laplacian(adj, T, dir=dir, selfloop=selfloop)
end

@doc raw"""
    scaled_laplacian(g[, T])

Scaled Laplacien matrix of graph `g`,
defined as ``\hat{L} = \frac{2}{\lambda_{max}} L - I`` where ``L`` is the normalized Laplacian matrix.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
"""
function scaled_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj))
    @assert issymmetric(adj) "scaled_laplacian only works with symmetric matrices"
    E = eigen(Symmetric(Array(adj))).values
    T(2. / maximum(E)) * normalized_laplacian(adj, T) - I
end

function scaled_laplacian(g::AbstractGraph, T::DataType=eltype(g))
    adj = adjacency_matrix(g, T)
    scaled_laplacian(adj, T)
end

"""
    random_walk_laplacian(g[, T]; dir=:out)

Random walk normalized Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
function random_walk_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out)
    d = degrees(adj, dir=dir)
    inv_d = 1 ./ d
    replace!(inv_d, typemax(float(T)) => zero(float(T)))  # avoid degree to be zero
    P = Diagonal(inv_d) * adj
    SparseMatrixCSC(T.(I - P))
end

"""
    signless_laplacian(g[, T]; dir=:out)

Signless Laplacian matrix of graph `g`.

# Arguments

- `g`: should be a adjacency matrix, `FeaturedGraph`, `SimpleGraph`, `SimpleDiGraph` (from Graphs)
    or `SimpleWeightedGraph`, `SimpleWeightedDiGraph` (from SimpleWeightedGraphs).
- `T`: result element type of degree vector; default is the element type of `g` (optional).
- `dir`: direction of degree; should be `:in`, `:out`, or `:both` (optional).
"""
signless_laplacian(adj::AbstractMatrix, T::DataType=eltype(adj); dir::Symbol=:out) =
    degree_matrix(adj, T, dir=dir) + SparseMatrixCSC(T.(adj))
