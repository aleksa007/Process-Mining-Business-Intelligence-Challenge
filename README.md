#Process Mining Group 13 - 2020

An attempt to make a well-functioning Process Mining GitHub repo. 

The workflow will consist of two branches:
- master
- develop

We will be working on the _develop_ branch and merging with _master_ once it's tested and 'deployable'.

To set up the local repo do the following:

1. git clone https://github.com/sakce/process.git
2. cd process
3. Check that you are on master branch by: git branch
- If not, do: git checkout master
4. Now we want to set from which branch should your local branch be updated. And this is the remote/<branch>. Do this: git branch -u origin/master
5. Now we'll do the same for develop branch.
- git checkout develop
- git branch -u origin/develop
6. To check if they're being tracked correctly: git branch -vv
- This should print something like 
* develop ....... [origin/develop] <commit msg>
master  ....... [origin/master] <commit msg>

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

