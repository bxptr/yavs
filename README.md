# yavs
simple and fast vector database with browser and python bindings

yavs (yet another vector store) is the only vector database that enables efficient insertion and retrieval cross-platform (within the browser with Web Assembly and locally with Python bindings).
yavs is made for agents, retrieval-based apps, and more that require a small/medium vector store to be modified locally and client-side. the idea is that vector stores can be propagated locally and used
in apps completely within the browser, erasing the need for a host with substantial memory or disk or setting up a database with a troublesome provider. 

see local bindings in `python/`, where i've included an example as well. `src/` has the Rust codebase and WASM bindings that compile with `wasm-pack` to what you can see in `pkg/`. `examples/` has a client-side app that tests
in-memory creation, loading from a file, insertion, and query.

yavs works off of a custom binary protocol that roughly looks like:
```
+------------+--------------+---------------+--------------+---------------+
|  MAGIC (4) |    VERSION   |   N_RECORDS   |     DIM      |    RESERVED   |
+------------+--------------+---------------+--------------+---------------+
|      RECORD 1: ID (16)    |   EMBEDDING   | META_LEN (4) |    METADATA   |
+--------------------------------------------------------------------------+
|      ...                                                      ...        |
+--------------------------------------------------------------------------+
```
it's relatively rudimentary but should scale to a solid number of records. it is very much possible to extend this with an [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) implementation.

