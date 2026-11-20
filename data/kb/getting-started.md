---
id: getting-started
title: Getting started with Brightwave CI
---
## Connect a repository
Sign in at app.brightwave.example and choose **New project**. Brightwave CI connects to GitHub, GitLab and Bitbucket through an OAuth app, and you pick the repositories it may read. Each repository becomes one project.

## Add a pipeline file
Commit a file named `.brightwave.yml` to the root of the default branch. Brightwave CI starts a pipeline on every push and on every pull request that changes code. A minimal file has one job with a `runs-on` runner size and a list of `steps`.

## Watch the first run
Open the project to see the pipeline graph. Each job shows live logs, its runner size and how many build minutes it used. Failed steps are highlighted and can be re-run on their own without restarting the whole pipeline.
