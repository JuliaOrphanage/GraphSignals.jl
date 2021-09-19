# GraphSignals.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yuehhua.github.io/GraphSignals.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yuehhua.github.io/GraphSignals.jl/dev)
[![Build Status](https://travis-ci.org/yuehhua/GraphSignals.jl.svg?branch=master)](https://travis-ci.org/yuehhua/GraphSignals.jl)
[![coverage report](https://gitlab.com/JuliaGPU/GraphSignals.jl/badges/master/coverage.svg)](https://gitlab.com/JuliaGPU/GraphSignals.jl/commits/master)

A generic graph representation for combining graph signals (or features) and graph topology (or graph structure). It supports the graph structure defined in JuliaGraphs packages (i.e. LightGraphs and SimpleWeightedGraphs) and compatible with APIs in JuliaGraphs packages. Graph signals are usually features, including node feautres, edge features and graph features. Features are contained in arrays and CuArrays are supported via CUDA.jl.

## Example

```julia
julia> using GraphSignals, LightGraphs

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

julia> nf = rand(3, 7);

julia> fg = FeaturedGraph(ug, nf=nf)
ERROR: DimensionMismatch("number of nodes must match between graph (4) and node features (7)")
...
```

## APIs

### Graph-related APIs

* `graph`
* `node_feature`
* `edge_feature`
* `global_feature`
* `mask`
* `has_graph`
* `has_node_feature`
* `has_edge_feature`
* `has_global_feature`
* `nv`
* `ne`
* `adjacency_list`
* `is_directed`
* `fetch_graph`

### Linear algebraic APIs

* `adjacency_matrix`
* `degrees`
* `degree_matrix`
* `inv_sqrt_degree_matrix`
* `laplacian_matrix`, `laplacian_matrix!`
* `normalized_laplacian`, `normalized_laplacian!`
* `scaled_laplacian`, `scaled_laplacian!`
