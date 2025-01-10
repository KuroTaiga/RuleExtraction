#create conda env
conda create --name ruleExtraction python=3.10 -y
conda activate ruleExtraction

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


pip install -r requirements.txt

#clone yolov7
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip install -r requirements.txt
cd ..

#get the weights
