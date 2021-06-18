# Changelog

All notable changes to this project will be documented in this file.

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
