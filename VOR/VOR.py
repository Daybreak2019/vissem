
import Train
import classify
import cosine_nlm
import cv2
import os
import classify
import numpy as np

def search_files(data_dir):
    
    search_query = input("Enter your query: ")

    files = os.listdir(data_dir)
    count = 0
    
    for file in files:
      path = os.path.join(data_dir, file)
      image = cv2.imread(path)
      labels = classify.classifier_classify(image)
      text1 = []
      for label in labels:
        text1.append("This program produces " + label) 
        
      cosines = []
      for text in text1:
        vector1 = cosine_nlm.text_to_vector(text)
        vector2 = cosine_nlm.text_to_vector(search_query)

        cosines.append(cosine_nlm.get_cosine(vector1, vector2))
      
      
      for cosine in cosines:
         if cosine>=0.5 : 
           count += 1
           print(count)
           print("programs found that produces output as" )
           print(search_query)
           #Also display path here
           break
                     
    if count == 0:
      print("No program found that matches your query!")



if __name__ == '__main__':
    #Train.classifier_train()
    search_files("./repo")
           

# Run: python classifier.py



