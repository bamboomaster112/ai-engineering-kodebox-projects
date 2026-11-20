---
id: runners
title: Hosted and self-hosted runners
---
## Hosted runner sizes
Hosted runners are fresh Linux virtual machines that are deleted after each job. Choose a size with `runs-on`:
- `small`: 2 vCPU and 4 GB of memory (the default)
- `medium`: 4 vCPU and 8 GB of memory, billed at 2x build minutes
- `large`: 8 vCPU and 16 GB of memory, billed at 4x build minutes

## Self-hosted runners
Install the Brightwave runner agent on your own machine to run jobs inside your network. Self-hosted runners do not use build minutes. Give each runner labels such as `gpu` or `arm64`, and target them with `runs-on: [self-hosted, gpu]`.

## Concurrency
Your plan limits how many jobs run at the same time across the organization. Jobs above the limit wait in a queue and start in order as slots free up.
