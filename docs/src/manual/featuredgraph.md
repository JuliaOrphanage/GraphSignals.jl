# FeaturedGraph

## Construct a FeaturedGraph and graph representations

A `FeaturedGraph` is aimed to represent a composition of graph representation and graph signals. A graph representation is required to construct a `FeaturedGraph` object. Graph representation can be accepted in several forms: adjacency matrix, adjacency list or graph representation provided from JuliaGraphs.

```julia
julia> adj = [0 1 1;
              1 0 1;
              1 1 0]
3×3 Matrix{Int64}:
 0  1  1
 1  0  1
 1  1  0

julia> FeaturedGraph(adj)
FeaturedGraph(
	Undirected graph with (#V=3, #E=3) in adjacency matrix,
)
```

Currently, `SimpleGraph` and `SimpleDiGraph` from LightGraphs.jl, `SimpleWeightedGraph` and `SimpleWeightedDiGraph` from SimpleWeightedGraphs.jl, as well as `MetaGraph` and `MetaDiGraph` from MetaGraphs.jl are supported.

If a graph representation is not given, a `FeaturedGraph` object will be regarded as a `NullGraph`. A `NullGraph` object is just used as a special case of `FeaturedGraph` to represent a null object.

```julia
julia> FeaturedGraph()
NullGraph()
```

### FeaturedGraph constructors

```@docs
NullGraph()
```

```@docs
FeaturedGraph
```

## Graph Signals

Graph signals is a collection of any signals defined on a graph. Graph signals can be the signals related to vertex, edges or graph itself. If a vertex signal is given, it is recorded as a node feature in `FeaturedGraph`. A node feature is stored as the form of generic array, of which type is `AbstractArray`. A node feature can be indexed by the node index, which is the same index for given graph.

Node features can be optionally given in construction of a `FeaturedGraph`.

```julia
julia> fg = FeaturedGraph(adj, nf=rand(5, 3))
FeaturedGraph(
	Undirected graph with (#V=3, #E=3) in adjacency matrix,
	Node feature:	ℝ^5 <Matrix{Float64}>,
)

julia> has_node_feature(fg)
true

julia> node_feature(fg)
5×3 Matrix{Float64}:
 0.534928  0.719566  0.952673
 0.395465  0.268515  0.335446
 0.79428   0.18623   0.454377
 0.530675  0.402474  0.00920068
 0.642556  0.719674  0.772497
```

Users check node/edge/graph features are available by `has_node_feature`, `has_edge_feature` and `has_global_feature`, respectively, and fetch these features by `node_feature`, `edge_feature` and `global_feature`.

### Getter methods

```@docs
graph
```

```@docs
node_feature
```

```@docs
edge_feature
```

```@docs
global_feature
```

### Check methods

```@docs
has_graph
```

```@docs
has_node_feature
```

```@docs
has_edge_feature
```

```@docs
has_global_feature
```

## Graph properties

`FeaturedGraph` is itself a graph, so we can query some graph properties from a `FeaturedGraph`.

```
julia> nv(fg)
3

julia> ne(fg)
3

julia> is_directed(fg)
false
```

Users can query number of vertex and number of edge by `nv` and `ne`, respectively. `is_directed` checks if the underlying graph is a directed graph or not.

### Graph-related APIs

```@docs
nv
```

```@docs
ne
```

```@docs
is_directed
```

## Pass `FeaturedGraph` to CUDA

Passing a `FeaturedGraph` to CUDA is easy. Just pipe a `FeaturedGraph` object to `cu`.

```
julia> using CUDA

julia> fg = fg |> cu
FeaturedGraph(
	Undirected graph with (#V=3, #E=3) in adjacency matrix,
	Node feature:	ℝ^5 <CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}>,
)
```

Or you can use `gpu` provided by Flux.

```
julia> using Flux

julia> fg = fg |> gpu
FeaturedGraph(
	Undirected graph with (#V=3, #E=3) in adjacency matrix,
	Node feature:	ℝ^5 <CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}>,
)
```

## Linear algebra for `FeaturedGraph`

`FeaturedGraph` supports the calculation of graph Laplacian matrix in inplace manner.

```
julia> fg = FeaturedGraph(adj, nf=rand(5, 3))
FeaturedGraph(
	Undirected graph with (#V=3, #E=3) in adjacency matrix,
	Node feature:	ℝ^5 <Matrix{Float64}>,
)

julia> laplacian_matrix!(fg)
FeaturedGraph(
	Undirected graph with (#V=3, #E=3) in Laplacian matrix,
	Node feature:	ℝ^5 <Matrix{Float64}>,
)

julia> laplacian_matrix(fg)
3×3 SparseArrays.SparseMatrixCSC{Int64, Int64} with 9 stored entries:
 -2   1   1
  1  -2   1
  1   1  -2
```

`laplacian_matrix!` mutates the adjacency matrix into a Laplacian matrix in a `FeaturedGraph` object and the Laplacian matrix can be fetched by `laplacian_matrix`. The Laplacian matrix is cached in a `FeaturedGraph` object and can be passed to a graph neural network model for training or inference. This way reduces the calculation overhead for Laplacian matrix during the training process.

`FeaturedGraph` supports not only Laplacian matrix, but also normalized Laplacian matrix and scaled Laplacian matrix calculation.

### Inplaced linear algebraic APIs

```@docs
laplacian_matrix!
```

```@docs
normalized_laplacian!
```

```@docs
scaled_laplacian!
```

### Linear algebraic APIs

Non-inplaced APIs returns a vector or a matrix directly.

```@docs
adjacency_matrix
```

```@docs
degrees
```

```@docs
degree_matrix
```

```@docs
laplacian_matrix
```

```@docs
normalized_laplacian
```

```@docs
scaled_laplacian
```
