pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /home/aistudio/data/
mkdir inference
unzip -o -d ./inference /home/aistudio/data/data10292/lits_test.zip

for zip in `ls ./inference/*.zip` ; do  unzip -o -d ./inference $zip ; done
rm ./inference/*.zip

mkdir preprocess
