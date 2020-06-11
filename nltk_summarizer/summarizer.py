
import nltk

with open("test.txt",'r') as testfile:
  sentences = testfile.readlines() 
 

#sentence = "The program produces barchart"
summary=[[]]
summary[0].append("The program produces ")
if len(sentences)>1:
  i=0
  for sent in sentences:
    if i==len(sentences)-2:
      break
    elif sent!= "":
      words = nltk.word_tokenize(sent)
      for j in range(3,len(words)-1):
        summary[0].append(words[j])
       
    summary[0].append(",")
    i+=1

  words = nltk.word_tokenize(sentences[i])
  summary[0].append("and")
  for j in range(3,len(words)-1):
    summary[0].append(words[j])
  summary[0].append(".")      
#      m=[]
#      m.append(nltk.pos_tag(words))
#      print(m)
#      print("\n")
else:
  words=nltk.word_tokenize(sentences[0])
  for i in range(3,len(words)-1):
    summary[0].append("")
    summary[0].append([words[i]])
  

print(*summary[0])

testfile.close()
