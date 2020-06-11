#ways to call linux commands
import os
import subprocess


# OR


#os.system('pwd')

def run_build(dir_path):
	#to run a make command
	#make [ -f makefile ] [ options ] ... [ targets ] ...
	path = os.path.join(dir_path)
    
	try:
		make_process = subprocess.call("cmake H:\Execution_engine\Execution_engine\programs\sqlitebrowser-master/", stderr=subprocess.STDOUT)
	except:
		perror("error occured during build")

if __name__ == '__main__':
    run_build("programs/geometric_shapes")