# scan_path='/home/aistudio/data/inference'
# label_path='/home/aistudio/data/inf_lab'

scan_path='/home/aistudio/data/scan_temp'
label_path='/home/aistudio/data/label_temp'

for name in `ls $scan_path`
do
  echo $name
  itksnap -g $scan_path/$name -s $label_path/$name
  # itksnap -g $scan_path/$name -s $label_path/test-segmentation${name:11}
done
