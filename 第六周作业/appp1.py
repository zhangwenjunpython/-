#班级：物理161 姓名：张文君
#编写时间：2019年4月4日19:00
#主要功能：随机产生50个数字，存一个列表中 list，然后从小到大排序，然后写入文件，
# 然后从文件中读取出来文件内容，然后反序，在追加到文件的下一行中
#说明D:\\woking file\\python\\存放路径为本机连接
#新创立的文件名为app1.txt
import random          #写入内容
random.seed(0)
list=[random.randint(0,100) for _ in range(50)]
list.sort()
fi=open("app1.txt",'w')
for i in list:
    fi.write(str(i)+",")     #数字顺序排列，每个数字用逗号隔开
fi.close()

#追加写内容
fo = open('app1.txt','r',encoding='utf-8')
f=fo.read().split(",")
f=f[::-1]
f=f[1:]                    #先读出文件，倒序排序
fo.close()

fire= open('app1.txt','a',encoding='utf-8')
fire.write("\n")
for line in f:
    fire.write(line+",")




  
