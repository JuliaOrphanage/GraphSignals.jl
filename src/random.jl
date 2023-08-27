"""
    random_walk(g, start, n=1)

Draw random walk samples from a given graph `g`. The weights for each edge on graph
are considered to be proportional to the transition probability.

# Arguments

- `g`: Data representing the graph topology. Possible type are
    - An adjacency matrix.
    - An `FeaturedGraph` or `SparseGraph` object.
- `start::Int`: A start point for a random walk on graph `g`.
- `n::Int`: Number of random walk steps.

# Usage

```julia
julia> using GraphSignals

julia> adjm = [0 1 0 1 1;
               1 0 0 0 0;
               0 0 1 0 0;
               1 0 0 0 1;
               1 0 0 1 0];

julia> fg = FeaturedGraph(adjm);

julia> random_walk(adjm, 1)
1-element Vector{Int64}:
 5

julia> random_walk(fg, 1, 3)
3-element Vector{Int64}:
 5
 4
 4

julia> using Flux

julia> fg = fg |> gpu;

julia> random_walk(fg, 4, 3)
3-element Vector{Int64}:
 1
 1
 1
```

See also [`neighbor_sample`](@ref)
"""
random_walk(A::AbstractMatrix, start::Int, n::Int=1) =
    [sample(1:size(A, 1), Weights(view(A, :, start))) for _ in 1:n]

random_walk(x::AbstractVector, n::Int=1) = [sample(1:length(x), Weights(x)) for _ in 1:n]

random_walk(sg::SparseGraph, start::Int, n::Int=1) = random_walk(sg.S, start, n)

random_walk(fg::FeaturedGraph, start::Int, n::Int=1) = random_walk(graph(fg), start, n)


"""
    neighbor_sample(g, start, n=1; replace=false)

Draw random samples from neighbors from a given graph `g`. The weights for each edge on graph
are considered to be proportional to the transition probability.

# Arguments

- `g`: Data representing the graph topology. Possible type are
    - An adjacency matrix.
    - An `FeaturedGraph` or `SparseGraph` object.
- `start::Int`: A vertex for a random neighbor sampling on graph `g`.
- `n::Int`: Number of random neighbor sampling.
- `replace::Bool`: Sample with replacement or not.

# Usage

```julia
julia> using GraphSignals

julia> adjm = [0 1 0 1 1;
               1 0 0 0 0;
               0 0 1 0 0;
               1 0 0 0 1;
               1 0 0 1 0];

julia> fg = FeaturedGraph(adjm);

julia> neighbor_sample(adjm, 1)
1-element Vector{Int64}:
 4

julia> neighbor_sample(fg, 1, 3)
3-element Vector{Int64}:
 5
 4
 2

julia> using Flux

julia> fg = fg |> gpu;

julia> neighbor_sample(fg, 4, 3, replace=true)
3-element Vector{Int64}:
 1
 5
 5
```

See also [`random_walk`](@ref)
"""
neighbor_sample(A::AbstractMatrix, start::Int, n::Int=1; replace::Bool=false) =
    sample(1:size(A, 1), Weights(view(A, :, start)), n; replace=replace)

neighbor_sample(x::AbstractVector, n::Int=1; replace::Bool=false) = sample(1:length(x), Weights(x), n; replace=replace)

neighbor_sample(sg::SparseGraph, start::Int, n::Int=1; replace::Bool=false) =
    neighbor_sample(sg.S, start, n; replace=replace)

neighbor_sample(fg::FeaturedGraph, start::Int, n::Int=1; replace::Bool=false) =
    neighbor_sample(graph(fg), start, n; replace=replace)
