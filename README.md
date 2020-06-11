A novel approach to program understanding based on visual semantics.

Project Structure:

The code has 4 modules:

 1.  VOR.py
 2.  Train.py
 3.  ImageExtract.py
 4.  Classify.py
 
 
VOR.py: 
	This is the main program which takes user query as input (for search). The program then extracts the information in the database. 
	Currently, the database is just a folder containing image files (visual outputs of programs). The images are fed to the classifier 
	(classify.py) to obtain object labels. The labels are fitted into fixed templates to generate sentences as image descriptions. Then 
	The sentences are matched with the user query.
	
Train.py
    This file includes code for CNN (Visual object recognizer) training. The program simply works by training Lenet with predefined class 
	labels (supervised). The training data is set of images stored in the folders. The folder names are the class labels to which the images
	belong.
ImageExtract.py
	This file includes code for image preprocessing before being fed to the classifier. The major task is to segment the input image into
	ROI. The subtasks involve converting the image to grayscale, applying threshold and finding contours and bounding boxes. The program 
	returns the segments(bounding boxes) as separate images to be classified by the classifier later.
	
Classify.py
	This code performs prediction of image labels based on the learned model.

How to run?

Simply run the program VOR.py (inside VOR folder)

This project is created in visual studio. Running the project with visual studio is recommended provided that all the dependencies (such as 
opencv) are installed and paths are correctly set.

The program VOR.py has two methods - Train.classifier_train() and search_files(). The train function call is used only once for training
Once the training is complete, the function call Train.classifier_train() can be commented out. Currently, the program is not designed to 
accomodate the appropriate actions (train or search) via command line arguments. It can be done with minor change in the code.

Other files in the project:

1. cosine_nlm.py is for matching user query against and program descriptions based on cosine similarity
2. Execution_engine.py is incomplete (code under test) for implementing automated building and execution of programs.
3. github_miner.py and gitminer.py are initial implementations for mining/downloading sample programs from github.
4. summarizer.py (inside folder nltk_summarizer) is standalone program (yet to be integrated with main project) for summarizing 
	multiple-sentences (program functionality descriptions) into fewer concise sentences). It is preliminary naive implementation
	based on nltk.
5. The folder "data" contains training and testing data. There are two folder with name "data". The folder inside "VOR" is actually 
	used, whereas the "data" folder outside the folder "VOR" is backup copy.

About the branch "WholeImageWithGensimNLM":

This branch is created to handle the whole-image-only kind of data and it also involves implementation of gensim-based NLM technique. The 
difference is that this version does not perform segmentation of the images and feeds whole images directly to the classifier. Whole images could be handled by modifying the code in the master 
branch. However, to avoid longer time in code modification, I created the branch. Also, this version uses gensim technique for NLM instead of naive
cosine similarity.