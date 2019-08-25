# face-recognition
1-Objective:
Objective of this project is to recognize a person through his face provided that one of these pictures of his face is in our database.
In other words, if a person "x" appear in a video and his picture exists in our database our application must know it and say this person name is "x".
2-Modification:
You must add 2 empty folder.
First folder "List-criminel" needed to add faces to the database.
Second folder "Trainning" is for file Training.py for create a training.yml.
3-How files work:
i)list_criminel.py-> With this file we detect the person's face and then we capture it.After that, we transform the captured image into Grayscale and we add it to the empty folder list-criminel.
ii)trainning.py-> This file help up to get the pictures and the label data ,convert to PIL image and form the model using faces and identifiers.
iii)reconnaissance_criminel.py->This file will detect the face in the video and compare it with the faces that are in our database.
  
  
  Thank u   
