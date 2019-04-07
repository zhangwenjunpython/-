#请尝试用大家学过的数据序列，生成一个CSV文件，在Excel中打开的结果是有年龄岁数成绩
#编写人：张文君，编写时间：2019月4月7月



import pandas as pd #任意的多组列表
a = [1,2,3,4,5] 
b=['mayi','jack','tom','rain','hanmeimei']
c=[18,21,23,25,26]
d=[99,86,83,94,81]
dataframe = pd.DataFrame({'NO.':a,'Name':b,'Age':c,'Score':d}) 
#将DataFrame存储为csv,index表示是否显示行名，
dataframe.to_csv("appp2.csv",index=False)

