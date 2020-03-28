cd /home/aistudio/data/
mkdir zip_temp
for zip in `ls ./data10273/*.zip` ; do unzip -o -d ./zip_temp $zip  ; done
# rm ./data10273 -rf # 不然空间可能不够
mkdir volume label
mv ./zip_temp/volume* ./volume
mv ./zip_temp/segmentation* ./label
for zip in `ls ./volume/*.zip` ; do unzip -o -d ./volume $zip; done
for zip in `ls ./label/*.zip` ; do  unzip -o -d ./label $zip ; done
rm ./volume/*.zip
rm ./label/*.zip
rm -rf zip_temp
# for seg in `ls ./label/`; do gzip $seg; done
# for vol in `ls ./volume/`; do gzip $vol; done
mkdir preprocess
# python ~/work/preprocess.py
# rm -rf ~/data/label/
# rm -rf ~/data/volume/
