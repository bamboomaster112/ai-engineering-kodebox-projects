---
id: caching-artifacts
title: Caching dependencies and keeping artifacts
---
## Dependency cache
Use the `cache` step with a key, such as a hash of your lock file, to restore dependencies between runs. Each project can store up to 5 GB of cache. When a project goes over the limit, the least recently used cache entries are deleted first.

## Artifacts
Artifacts are files a job uploads for later jobs or for people to download, such as test reports or build outputs. By default artifacts are kept for 30 days. Team plans can raise retention to 90 days and Enterprise plans to 400 days with `retention-days`.

## Cleaning up
Delete caches from **Project settings > Caches** if a corrupted cache keeps breaking builds. Deleting a cache is safe: the next run simply rebuilds it.
