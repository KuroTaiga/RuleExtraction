#create conda env
conda create -n ruleExtraction python=3.10 -y
conda activate ruleExtraction

pip install -r requirements.txt

#clone yolov7
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt
cd ..

#get the weights
