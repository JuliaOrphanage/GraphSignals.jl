# Changelog

All notable changes to this project will be documented in this file.

## [0.5.2]

- make `kneighbors_graph` non-differentiable and enable batch learning

## [0.5.1]

- fix invalid `setfield!` for `CuSparseMatrixCSC`

## [0.5.0]

- add `kneighbors_graph`
- fix doc test and some bug

## [0.4.3]

- Resolve conflict with StatsBase

## [0.4.2]

- support `collect(edges(g))` for graph without edges

## [0.4.1]

- add `has_all_self_loops` and `isneighbors`

## [0.4.0]

- add `SparseSubgraph`
- relax `FeaturedGraph`
- drop support of `edge_scatter`, `neighbor_scatter`, `repeat_nodes` and `promote_graph`

## [0.3.13]

- fix collect for edges

## [0.3.12]

- update docs

## [0.3.11]

- inplace normalized adjacency matrix
- specify type for `FeaturedGraph`

## [0.3.10]

- add `FeaturedSubgraph` for subgraphing `FeaturedGraph`

## [0.3.9]

- add sample for generate random subgraph for `FeaturedGraph` and `FeaturedSubgraph`
- drop `cpu_neighbors` and `cpu_incident_edges`
- add `neighbors`, `incident_edges`, `repeat_nodes` for `FeaturedGraph` and `FeaturedSubgraph`
- add `parent` for `FeaturedGraph` and `FeaturedSubgraph`
- access features in `FeaturedSubgraph` change to `FeaturedGraph`

## [0.3.8]

- bug fix for message-passing network

## [0.3.7]

- correct edge direction for `has_edge` and `edge_index`
- add `has_edge` for `FeaturedGraph`

## [0.3.6]

- add random walk and neighbor sampling on graph
- add `edges` and `neighbors` as API for `FeaturedGraph`
- add `cpu_neighbors`
- add `collect` and `sparse` for `SparseGraph`

## [0.3.5]

- fix normalized_adjacency_matrix for FeaturedGraph

## [0.3.4]

- correct degree_matrix with CuSparseMatrixCSC input
- correct normalized adjacency matrix with self-loop
- adjacency_matrix returns dense arrays

## [0.3.3]

- add normalized adjacency matrix
- migrate GraphLaplacians
- migrate LightGraphs to Graphs

## [0.3.2]

- migrate LightGraphs to Graphs

## [0.3.1]

- change function name `check_num_node`, `check_num_edge` into `check_num_nodes`, `check_num_edges`

## [0.3.0]

- `FeaturedGraph` use `SparseGraph` as core graph structure
- `SparseGraph` support cuda for undirected graph and directed graph
- make `FeaturedGraph` idempotent
- make `FeaturedGraph` can be pass to gpu
- add scatter for FeaturedGraph
- update manual
- drop `fetch_graph`

## [0.2.3]

- Support gradient for neighbor_scatter and edge_scatter
- Support CUDA up to v3.3

## [0.2.2]

- EdgeIndex support adjacency matrix and juliagraphs
- add `neighbor_scatter` for scatter across neighbors

## [0.2.1]

- Support CUDA up to v3.2
- FeaturedGraph checks feature dimensions in constructors and before setting property
- add Base.show for FeaturedGraph
- add EdgeIndex and edge_scatter for providing an indexing strucutre to message-passing
- add GraphMask for masking FeaturedGraph

## [0.2.0]

- Support Julia v1.6 and CUDA v2.6 only

## [0.1.13]

- Support CUDA v2.6
- Support FillArrays v0.11
- Support Zygote v0.6

## [0.1.12]

- Dimension check should be done in layer but previous to computation

## [0.1.11]

- Refactor checking feature dimensions
- Support Julia v1.6
- Fix CuArray promotion

## [0.1.10]

- Bug fix

## [0.1.9]

- Add feature dimension check on constructor
- Refactor FeaturedGraph constructor

## [0.1.8]

- Accept transposed matrix as node feature
- Refactor

## [0.1.7]

- Support ne for adjacency list

## [0.1.6]

- Support is_directed for FeaturedGraph
- Fix test case
- Bug fix

## [0.1.5]

- Add ne for adjacency list

## [0.1.4]

- Take off dimension check

## [0.1.3]

- Bug fix

## [0.1.2]

- Laplacian matrix can be contained in FeaturedGraph
- Add dimensional check among graph, node and edge feature
- Add ne for matrix
- Add mask for FeaturedGraph
- Add fetch_graph
