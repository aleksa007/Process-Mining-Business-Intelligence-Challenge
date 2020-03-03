An attempt to make a well-functioning Process Mining GitHub repo. 

The workflow will consist of two branches:
- master
- develop

We will be working on the develop branch and merging with master once it's tested and 'deployable'.

"The master branch is meant to be stable, and it is the social contract of
open source software to never, ever push anything to master that
is not tested, or that breaks the build. "


"Very important: when merging, we need to be on the branch
that we want to merge to. Basically, we will be telling git,
“See that new thing? It’s ok to bring it over here now.” "

Step : git merge <branch_name> –no-ff

From remote/master to origin/master
- git push

To delete a merged branch: git branch -d <branch_name> 
No worries: if you accidentally attempt to delete a branch that has not yet been merged, git will throw an error.



"Because, remember: Don’t. Mess. With. The. Master. "

