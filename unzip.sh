ls /home/aistudio/data/data10273/*.zip | xargs -n1 unzip -d /home/aistudio/data/zip_temp -o
cd ~/data/
mkdir volume label
mv ./zip_temp/volume* ./volume
mv ./zip_temp/segmentation* ./label
find ./volume/*.zip | xargs -n1 unzip -d ./volume
find ./label/*.zip | xargs -n1 unzip -d ./label
rm ./volume/*.zip
rm ./label/*.zip
rm -rf zip_temp
python ~/work/preprocess.py
# rm -rf ~/data/label/
# rm -rf ~/data/volume/

export CPU_NUM=1