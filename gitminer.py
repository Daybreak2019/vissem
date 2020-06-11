""" Reference taken from https://github.com/dustin/py-github

	https://pygithub.readthedocs.io/en/latest/github.html#github.MainClass.Github.get_repos
https://developer.github.com/v3/apps/available-endpoints/

https://github.com/PyGithub/PyGithub
https://stackoverflow.com/questions/10625190/most-suitable-python-library-for-github-api-v3
"""
from github import Github
import github.GithubObject

gh = github.Github()

#repository list for a user
"""print("My repo names:")
for r in gh.repo.forUser(me):
    print(r.name)
"""
#search for a repository

#for r in gh.repos.search('memcached'):
 #   print("%s's %s" % (r.username, r.name))

cnt = 0

for repo in gh.get_repos(since=2018):

    cnt += 1
    print(str(cnt) + " :" + repo.name)


#for repo in gh.get_user("shivrajpant").get_repos():
#   print(repo.name)