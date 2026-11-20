---
id: pipeline-syntax
title: Pipeline syntax reference
---
## Stages and jobs
A pipeline is a set of jobs. Jobs run in parallel unless one lists another under `needs`, which makes it wait for that job to succeed. Use `when: manual` for a job that waits for someone to press **Run**, such as a production deploy.

## Matrix builds
`matrix` runs one job definition several times with different values, for example three Python versions on two operating systems. Each combination counts as its own job for concurrency and build minutes.

## Timeouts
Every job has a timeout. The default is 60 minutes. You can raise it with `timeout-minutes` up to 6 hours on the Team plan and 24 hours on Enterprise. The Free plan caps jobs at 60 minutes.

## Retries
Add `retry: 2` to a job to rerun it automatically up to two more times when it fails. Retries help with flaky tests, but each attempt uses build minutes.
