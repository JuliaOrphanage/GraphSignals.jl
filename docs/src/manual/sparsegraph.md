# Sparse graph Strucutre

## The need of graph structure

Graph convolution can be classified into spectral-based graph convolution and spatial-based graph convolution. Spectral-based graph convolution relys on the algebaric operations, including `+`, `-`, `*`, which are applied to features with graph structure. Spatial-based graph convolution relys on the indexing operations, since spatial-based graph convolution always indexes the neighbors of vertex. A graph structure can be use under two view point of a part of algebaric operations or an indexing structure.

Message-passing neural network requires to access neighbor information for each vertex. Messages are passed from a vertex's neighbors to itself. A efficient indexing data structure is required to access incident edges or neighbor vertices from a specific vertex.

## `SparseGraph`

SparseGraph is implemented with sparse matrix. It is built on top of built-in sparse matrix, `SparseMatrixCSC`. `SparseMatrixCSC` can be used as a regular matrix and performs algebaric operations with matrix or vectors.

To benefit message-passing scheme, making a graph structure as an indexing structure is important. A well-designed indexing structure is made to leverage the sparse format of `SparseMatrixCSC`, which is in CSC format. CSC format stores sparse matrix in a highly compressed manner. Comparing to traditional COO format, CSC format compresses the column indices into column pointers. All values are stored in single vector. If we want to index the sparse matrix `A`, the row indices can be fetched by `rowvals[colptr[j]:(colptr[j+1]-1)]` and the non-zero values can be indexed by `nzvals[colptr[j]:(colptr[j+1]-1)]`. The edge indices are designed in the same manner `edges[colptr[j]:(colptr[j+1]-1)]`. This way matches the need of indexing neighbors of vertex. This makes neighbor indices or values close together. It takes ``O(1)`` to get negihbor indices, instead of searching neighbor in ``O(N)``. Thus, `SparseGraph` takes both advantages of both algebaric operations and indexing operations.

## Create `SparseGraph`

`SparseGraph` accepts adjacency matrix, adjacency list, and almost all graphs defined in JuliaGraphs.

```julia
julia> using GraphSignals, LightGraphs

julia> ug = SimpleGraph(4)
{4, 0} undirected simple Int64 graph

julia> add_edge!(ug, 1, 2); add_edge!(ug, 1, 3); add_edge!(ug, 1, 4);

julia> add_edge!(ug, 2, 3); add_edge!(ug, 3, 4);

julia> sg = SparseGraph(ug)
SparseGraph(#V=4, #E=5)
```

The indexed adjacency list is a list of list strucutre. The inner list consists of a series of tuples containing a vertex index and a edge index, respectively.

## Operate `SparseGraph` as graph

It supports basic graph APIs for querying graph information, including number of vertices `nv` and number of edges `ne`.

```julia
julia> is_directed(sg)
false

julia> nv(sg)
4

julia> ne(sg)
5

julia> eltype(sg)
Int64

julia> has_vertex(sg, 3)
true

julia> has_edge(sg, 1, 2)
true
```

We can compare two graph structure if they are equivalent or not.

```julia
julia> adjm = [0 1 1 1; 1 0 1 0; 1 1 0 1; 1 0 1 0]
4×4 Matrix{Int64}:
 0  1  1  1
 1  0  1  0
 1  1  0  1
 1  0  1  0

julia> sg2 = SparseGraph(adjm, false)
SparseGraph(#V=4, #E=5)

julia> sg == sg2
true
```

We can also iterate over edges.

```julia
julia> for (i, e) in edges(sg)
           println("edge index: ", i, ", edge: ", e)
       end
edge index: 1, edge: (2, 1)
edge index: 2, edge: (3, 1)
edge index: 3, edge: (3, 2)
edge index: 4, edge: (4, 1)
edge index: 5, edge: (4, 3)
```

Edge index is the index for each edge. It is used to index edge features.

## Indexing operations

To get neighbors of a specified vertex, `neighbors` is used by passing a `SparseGraph` object and a vertex index. A vector of neighbor vertex index is returned.

```julia
julia> neighbors(sg, 1)
3-element view(::Vector{Int64}, 1:3) with eltype Int64:
 2
 3
 4
```

To get incident edges of a specified vertex, `incident_edges` can be used and it will return edge indices.

```julia
julia> incident_edges(sg, 1)
3-element view(::Vector{Int64}, 1:3) with eltype Int64:
 1
 2
 4
```

An edge index can be fetched by querying an edge, for example, edge `(1, 2)` and edge `(2, 1)` refers to the same edge with index `1`.

```julia
julia> edge_index(sg, 1, 2)
1

julia> edge_index(sg, 2, 1)
1
```

One can have the opportunity to index the underlying sparse matrix.

```julia
julia> sg[1, 2]
1

julia> sg[2, 1]
1
```

## Aggregate over neighbors

In message-passing scheme, it is always to aggregate node features or edge feature from neighbors. For convention, `edge_scatter` and `neighbor_scatter` are used to apply aggregate operations over edge features or neighbor vertex features. The actual aggregation is supported by `scatter` operations.

```julia
julia> nf = rand(10, 4);

julia> neighbor_scatter(+, nf, sg)
10×4 Matrix{Float64}:
 1.54937   1.03974   1.72926  1.03974
 1.38554   0.775991  1.34106  0.775991
 1.13192   0.424888  1.34657  0.424888
 2.23452   1.63226   2.436    1.63226
 0.815662  0.718865  1.25237  0.718865
 2.35763   1.42174   2.26442  1.42174
 1.94051   1.44812   1.71694  1.44812
 1.83641   1.89104   1.80857  1.89104
 2.43027   1.92217   2.37003  1.92217
 1.58177   1.16149   1.87467  1.16149
```

For example, `neighbor_scatter` aggregates node features `nf` via neighbors in graph `sg` by `+` operation.

```julia
julia> ef = rand(9, 5);

julia> edge_scatter(+, ef, sg)
9×4 Matrix{Float64}:
 2.22577  0.967172  1.92781   1.92628
 1.4842   1.20605   2.30014   0.849819
 2.20728  1.01527   0.899094  1.35062
 1.09119  0.589925  1.62597   1.51175
 1.42288  1.63764   1.23445   0.693258
 1.57561  0.926591  1.72599   0.690108
 1.68402  0.544808  1.58687   1.70676
 1.10908  1.0898    1.05256   0.508157
 2.33764  1.26419   1.87927   1.11151
```

Or, `edge_scatter` aggregates edge features `ef` via incident edges in graph `sg` by `+` operation.

## `SparseGraph` APIs

```@docs
SparseGraph
GraphSignals.neighbors
incident_edges
neighbor_scatter
edge_scatter
```

## Internals

In the design of `SparseGraph`, it resolve the problem of indexing edge features. For a graph, edge is represented in `(i, j)` and edge features are considered as a matrix `ef` with edge number in its column. The problem is to unifiedly fetch corresponding edge feature `ef[:, k]` for edge `(i, j)` over directed and undirected graph. To resolve this issue, edge index is set to be the unique index for each edge. Further, `aggregate_index` is designed to generate indices for aggregating from neighbor nodes or incident edges. Conclusively, it provides the core operations needed in message-passing scheme.
