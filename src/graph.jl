"""
    adjacency_list(adj)

Transform a adjacency matrix into a adjacency list.
"""
function adjacency_list(adj::AbstractMatrix{T}) where {T}
    n = size(adj,1)
    @assert n == size(adj,2) "adjacency matrix is not a square matrix."
    A = (adj .!= zero(T))
    if !issymmetric(adj)
        A = A .| A'
    end
    indecies = collect(1:n)
    ne = Vector{Int}[indecies[view(A, :, i)] for i = 1:n]
    return ne
end

adjacency_list(adj::AbstractVector{<:AbstractVector{<:Integer}}) = adj
adjacency_list(g::AbstractGraph) = Vector{Int}[outneighbors(g, i) for i = 1:nv(g)]

GraphSignals.nv(g::AbstractMatrix) = size(g, 1)
nv(g::AbstractVector{T}) where {T<:AbstractVector} = size(g, 1)

function GraphSignals.ne(g::AbstractMatrix, directed::Bool=is_directed(g))
    g = Matrix(g) .!= 0

    if directed
        return sum(g)
    else
        return div(sum(g) + sum(diag(g)), 2)
    end
end

function ne(g::AbstractVector{T}, directed::Bool=is_directed(g)) where {T<:AbstractVector}
    if directed
        return sum(length, g)
    else
        e = 0
        for i in 1:length(g)
            for j in g[i]
                if i ≤ j
                    e += 1
                end
            end
        end
        return e
    end
end

GraphSignals.is_directed(g::AbstractMatrix) = !issymmetric(Matrix(g))

function is_directed(g::AbstractVector{T}) where {T<:AbstractVector}
    edges = Set{Tuple{Int64,Int64}}()
    for (i, js) in enumerate(g)
        for j in Set(js)
            if i != j
                e = (i,j)
                if e in edges
                    pop!(edges, e)
                else
                    push!(edges, (j,i))
                end
            end
        end
    end
    !isempty(edges)
end
