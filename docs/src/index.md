```@meta
CurrentModule = GraphSignals
```

# GraphSignals

GraphSignals is aim to provide a general data structure, which is composed of a graph and graph signals, in order to support a graph neural network library, specifically, GeometricFlux.jl. The concept of graph is used ubiquitously in several fields, including computer science, social science, biological science and neural science. GraphSignals provides graph signals attached to a graph as a whole for training or inference a graph neural network. Some characteristics of this package are listed:

- Graph signals can be node features, edge features or global features, which are all general arrays.
- Graph Laplacian and related matrices are supported to calculated from a general data structure.
- Support graph representations from JuliaGraphs.

## Example

`FeaturedGraph` supports various graph representations.

It supports graph in adjacency matrix.

```julia
julia> adjm = [0 1 1 1;
               1 0 1 0;
               1 1 0 1;
               1 0 1 0];

julia> fg = FeaturedGraph(adjm)
FeaturedGraph(
	Undirected graph with (#V=4, #E=5) in adjacency matrix,
)
```

It also supports graph in adjacency list.

```julia
julia> adjl = [
               [2, 3, 4],
               [1, 3],
               [1, 2, 4],
               [1, 3]
               ];

julia> fg = FeaturedGraph(adjl)
FeaturedGraph(
	Undirected graph with (#V=4, #E=5) in adjacency matrix,
)
```

It supports `SimpleGraph` from LightGraphs and convert adjacency matrix into a Laplacian matrix as well.

```julia
julia> using LightGraphs

julia> N = 4
4

julia> ug = SimpleGraph(N)
{4, 0} undirected simple Int64 graph

julia> add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 1, 4);

julia> add_edge!(ug, 2, 3); add_edge!(ug, 3, 4);

julia> fg = FeaturedGraph(ug)
FeaturedGraph(
	Undirected graph with (#V=4, #E=5) in adjacency matrix,
)

julia> laplacian_matrix!(fg)
FeaturedGraph(
	Undirected graph with (#V=4, #E=5) in Laplacian matrix,
)
```

Features can be attached to it.

```julia
julia> N = 4
4

julia> E = 5
5

julia> nf = rand(3, N);

julia> ef = rand(5, E);

julia> gf = rand(7);

julia> fg = FeaturedGraph(ug, nf=nf, ef=ef, gf=gf)
FeaturedGraph(
	Undirected graph with (#V=4, #E=5) in adjacency matrix,
	Node feature:	ℝ^3 <Matrix{Float64}>,
	Edge feature:	ℝ^5 <Matrix{Float64}>,
	Global feature:	ℝ^7 <Vector{Float64}>,
)
```

If there are mismatched node features attached to it, a `DimensionMismatch` is throw out and hint user.

```julia
julia> nf = rand(3, 7);

julia> fg = FeaturedGraph(ug, nf=nf)
ERROR: DimensionMismatch("number of nodes must match between graph (4) and node features (7)")
```
