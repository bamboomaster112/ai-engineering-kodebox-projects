---
id: troubleshooting-builds
title: Troubleshooting failed builds
---
## Exit code 137
Exit code 137 means the job was killed because it ran out of memory. Use a bigger runner with `runs-on: medium` or `runs-on: large`, or reduce parallel work inside the job, for example with fewer test workers.

## No space left on device
Hosted runners have 30 GB of free disk. Clean build outputs between steps, avoid caching large folders, and delete old caches from the project settings.

## Jobs that hang or time out
A job that produces no log output for 20 minutes is cancelled as stalled. Long-running jobs should print progress. If a job hits its timeout, raise `timeout-minutes` within your plan's limit.

## Flaky tests
Use `retry` on the job while you fix the flaky test, and quarantine the test in your test runner so it does not block merges.
