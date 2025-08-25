git clone https://github.com/Megvii-BaseDetection/YOLOX

cd  YOLOX

install depedency in requirement txt



git clone https://github.com/risal098/AI-cctv-antiburglar.git

copy AI-cctv-antiburglar/tiny_gabung.py to YOLOX/exps/example/custom/

copy AI-cctv-antiburglar/tiny_gabung.pth to YOLOX/

copy AI-cctv-antiburglar/tiny_raspiapp.py to YOLOX/


now go to tiny_raspiapp.py, and change "capdict" variable based on your needs (the camera ip).

you can also change config variable based on your need

ready to run (python3 tiny_raspiapp.py)

