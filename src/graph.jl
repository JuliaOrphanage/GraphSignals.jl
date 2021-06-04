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


function adjacency_list(g::AbstractGraph)
    N = nv(g)
    Vector{Int}[outneighbors(g, i) for i = 1:N]
end

Zygote.@nograd adjacency_list

GraphSignals.nv(g::AbstractMatrix) = size(g, 1)
nv(g::AbstractVector{T}) where {T<:AbstractVector} = size(g, 1)

Zygote.@nograd nv

function GraphSignals.ne(g::AbstractMatrix; self_loop::Bool=false)
    g = Matrix(g .!= 0)

    if issymmetric(g)
        if self_loop
            return div(sum(g .+ diagm(diag(g))), 2)
        else
            return div(sum(g .- diagm(diag(g))), 2)
        end
    else
        if self_loop
            return sum(g)
        else
            return sum(g .- diagm(diag(g)))
        end
    end
end

function ne(g::AbstractVector{T}, directed::Bool=is_directed(g)) where {T<:AbstractVector}
    for i in 1:length(g)
        filter!(x -> x != i, g[i])
    end
    s = map(x -> count(x .== x), g)
    return directed ? sum(s) : div(sum(s), 2)
end

Zygote.@nograd ne

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

GraphSignals.is_directed(g::AbstractMatrix) = !issymmetric(g)

Zygote.@nograd is_directed
