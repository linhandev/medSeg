对数据进行预处理和后处理的一些有用的脚本，功能介绍在[../README.md]中有写

# 数据格式转换
写代码角度讲nii格式一般用起来比较方便。dcm包含的信息多，但是一个文件夹不一定就是一个序列；一层一个文件的保存方式下，进行大量文件读写I/O效率也不高。这个项目基本都用nii格式，记录一些格式转换方法。

[dcm2niix](https://github.com/rordenlab/dcm2niix)可以将dcm转换nii，下面是一个命令行转换的例子。
```shell
dcm2niix -f 输出文件名，支持填入多种扫描里的信息 -d 9 -c 在输出nii文件中写注释 dcm文件夹
```
train目录下的 [mhd2nii.py](./train/mhd2nii.py) 可以将一个目录下的mhd格式扫描转成nii。

# 数据标注
标注数据的时候ITK-snap用起来很方便，用好命令行参数可以自动进行文件打开，节省时间。
```shell
count=0 ;
tot=`ls -l | wc -l`
for f in `ls`;
do count=`expr $count + 1`;
echo $count / $tot;
echo $f;
echo -e "\n";
itksnap -s /path/to/label/${f} -g /path/to/scan/${f} --geometry 1920x1080+0+0;
done
```

beep函数可以用扬声器测试程序发出一声beep，结合定时可以更好掌握标注时间
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
count=`expr $count + 1`;
echo $count / `ls -l | wc -l`;
echo ${f};
echo -e "\n";

# 打开扫描和标签
itksnap -s ./${f} -g ../nii/${f} --geometry 1920x1080+0+0;

# 文件归档
# cp ./${f} ../manual-label/
mv ./${f} ../manual-fin../ished/

# 关闭没响的beep
kill -9 $pid2
kill -9 $pid4
kill -9 $pid6
kill -9 $pid8
done
```
