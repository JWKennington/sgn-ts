# How to contribute

Source code is hosted on https://git.ligo.org/greg/sgn-ts.  You can request membership at a level appropriate for development there.


The `sgn` team uses the standard git-branch-and-merge workflow, which has brief description
at [GitLab](https://docs.gitlab.com/ee/gitlab-basics/feature_branch_workflow.html) and a full description
at [BitBucket](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow). 


## Local Git Workflow

In general the steps for working with feature branches are
:  1. Create a new branch from master: `git checkout -b feature-short-desc`
   1. Edit code (and tests)
   1. Make sure that the top level makefile runs before any commits. It checks formatting, linting, typing and tests.  It should be sufficient to simply run `make`. It is expected that all code changes will also come with 100% code coverage for tests.
   1. Commit changes: `git commit . -m "comment"`
   1. Push branch: `git push --set-upstream origin feature-short-desc`
   1. Create merge request on GitLab

## Creating a Merge Request

Once you push feature branch, GitLab will prompt you the next time you visit the sgnts homepage with a message: Click “Create Merge Request”, or you can
also go to the branches page (Repository > Branches) and select “Merge Request” next to your branch.

When creating a merge request
:  1. Add short, descriptive title
   1. Add description
       - (Uses markdown .md-file style)
       - Summary of additions / changes
       - Describe any tests run (other than CI)
   1. Click “Create Merge Request”

The Overview page give a general summary of the merge request, including
:  1. Link to other page to view changes in detail (read below)
   1. Code Review Request
   1. Test Suite Status
   1. Discussion History
   1. Commenting

You may use any of the MR tools that are practical in order to provide and respond to feedback so long as you are attentive to requests with clear responses.
If threads or reviews have been started you will need to complete or resolve them in order to merge.


## Merging
:  1. Check all tests passed and that code coverage is 100%
   1. Check all review comments resolved
   1. Check that the librarian(s) has given the go ahead
   1. Before clicking “Merge”
       - Check “Delete source branch”
       - Check “Squash commits” if branch history not tidy, though it is preferred to leave meaningful history in place.
   1. Click “Merge”
