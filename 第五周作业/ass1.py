#编写人：物理161，张文君  编写时间：2019年3月29日
#功能：
#输入：
# 请输入日期1： 2019,3,29
# 请输入日期2： 2020,3,28
# 输出：
# 两个日期间隔XXX天。
#基本的思路是
#  建立一个古老的起始年例如公元0年1月1日，然后分别计算，两个时间距离这个公元年的天数，天数再相减，就知道这两个时间点相差多少了

star=input("请起始输入年月日按“，”隔开\n")
over=input("请终止输入年月日按“，”隔开\n")
star=star.split(",")         #分别切开成年月日
over=over.split(",")
star=[int(i) for i in star]
over=[int(i) for i in over]
a,b,c=star
x,y,z=over

#先生成一个大小月天数的字典，方便我们计算月跟月之间的天数
mon=[0,31,28,31,30,31,30,31,31,30,31,30,31]     #前面加个0是为了方便表示月份mon【1】直接表示1月份的天数


#做一个判断闰年的函数
def runyear(years):       #输入年份还回天数
   if(years%400==0): return 366
   if(years%100==0): return 365
   if(years%4==0):   return 366
   return 365

#生成一个输入年月日还回距离起始元年的天数
def days(Y,M,D):    
    day=0 
    for k in range(1,Y) :    #先算相差年的天数
        day+=runyear(k)
    for j in range(1,M):   #再计算相差月的天数
        if (j==2 and runyear(Y)==366):
            day+=29
        else:
            day+=mon[j]
    day+=D
    return(day)

print(days(x,y,z)-days(a,b,c))
