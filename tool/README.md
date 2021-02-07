Useful scripts for preprocess and postprocessing medical image data.

# 数据格式转换
对计算机处理来说nii一般是比较方便的。dcm包含的信息最多但是一个文件夹不一定就是一个序列，大量文件读写I/O效率也可能不高。Anyway本项目中都是用的nii格式的输入，所以记录一些其他格式转nii的方法。

dcm2niix可以方便的将dcm转换nii，下面是一个命令行转换的例子。
```shell
count=0 ;
total=`ls -l | wc -l`
for f in `ls`;
do count=`expr $count + 1`;
echo $count / $total;
dcm2niix -f $f -o ../nii_raw/ -c $f $f
echo -e "\n"
echo -e "\n"
done
```
train目录下的 mhd2nii.py 可以将一个目录下的mhd格式扫描转换成nii。

# 数据标注
标注数据的时候ITK-snap用起来很方便，用好命令行参数可以自动进行文件打开，节省时间。
```shell
count=0 ;
for f in `ls`;
do count=`expr $count + 1`;
echo $count / `ls -l | wc -l`;
echo ${f};
echo -e "\n";
itksnap -s ../label/${f} -g ${f} --geometry 1920x1080+0+0;
done
```

beep函数可以用扬声器测试程序发出三声beep，更好的掌握标注一个case的时间。
```shell
beep1()
{
  ( \speaker-test --frequency $1 --test sine )&
    pid=$!
    \sleep ${2}s
    \kill -9 $pid
}

beep()
{
        beep1 350 0.2
        beep1 350 0.2
        beep1 350 0.2
        beep1 350 0.4
        sleep 1
}

count=0 ;
for f in `ls`;
do
# 四个 beep
(sleep 2m; beep;) &
pid2=$!
(sleep 4m; beep; beep;) &
pid4=$!
(sleep 6m; beep; beep; beep;) &
pid6=$!
(sleep 8m; beep; beep; beep; beep; ) &
pid8=$!

# 计数
count=`expr $count + 1`;胡金萍_20201024230539946.nii.gz
echo $count / `ls -l | wc -l`;
echo ${f};
echo -e "\n";

# 打开扫描和标签
itksnap -s ./${f} -g ../nii/${f} --geometry 1920x1080+0+0;

# 文件归档
# cp ./${f} ../manual-label/
mv ./${f} ../manual-finished/

# 关闭没响的beep
kill -9 $pid2
kill -9 $pid4
kill -9 $pid6
kill -9 $pid8
done
```
