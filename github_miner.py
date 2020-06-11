
""" References 
https://media.readthedocs.org/pdf/pygithub/stable/pygithub.pdf
https://github.com/dustin/py-github
https://pygithub.readthedocs.io/en/latest/github.html#github.MainClass.Github.get_repos
https://developer.github.com/v3/apps/available-endpoints/

https://github.com/ishepard/pydriller

https://github.com/UnkL4b/GitMiner
"""
import time
from github import Github
import github.GithubObject

gh = github.Github()

f = open("output.txt","w")

def search_reppo():
	#search repositories with particular language and having cmake file
	
	for i in range (100):
		try:
			
			repositories = gh.search_repositories(query='language:java').get_page(i)
			for repo in repositories:
				if "build.xml" in repo.get_contents(""):
					
					url = repo.get_archive_link('zipball')
					f.write(repo.name + "   " + url + "\n")

		except: 
			print("The search API hit the ratelimit, going sleep for 1 hour")
			time.sleep(3601)
			self.search_reppo()
			
"""

def search_reppo(self, interval):
	while True:
		try:
			it = enumerate(self.gh.search_repositories(query="java" + interval))
			yield from it
			return   

		except:  
			print.warning("Going to sleep for 1 hour. The search API hit the limit")
			time.sleep(3600)
			it = self.search(interval)
	return it


def 
"""
"""
def download_reppo():
	url = 

	#get list of first page (paginated list of 100) of repositories
	cnt = 0
	for repo in gh.get_repos().get_page(0):
		cnt += 1
		print(str(cnt) + " :" + repo.name)


#get the archive link of repositories

for repo in gh.get_repos().get_page(0):
	print(repo.get_archive_link('zipball'))

"""
#for repo in gh.get_user("shivrajpant").get_repos():
#   print(repo.name)

if __name__ == '__main__':
	search_reppo()
