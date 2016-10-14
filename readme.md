Under working:Programs for process cryo-em images<br>
=====
mrcProcess:adjust the images,enhance the contrast and do GaussBlur;convert the images into matrix<br><br>
mrcParser:parser the mrcstar file;read mrcstar head and data;put data in the vector<br><br>
mnistLoad:create mnist-format data to run CNN  through caffe or other tools<br><br>
recent goals(before 9.5)<br>
-------
1.write app/mrcProcess.cpp<br>
2.write app/createMnistTrainSet.cpp<br>
  write app/createMnistTestSet.cpp>br>
  add function "Mnistlabe" lto libMnist<br>

before 9.30 goals(mission complete-9.21)<br>
---------------------
1.pick data from mrcstar and save them in a buffer,write buffer into the MNISTfile while buffer is full.
