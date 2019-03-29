# 编写人：物理161张文君，编写时间：2019年3月29号
#功能如下：
# 菜单
# 1.	新增联系人. (重名)
# 2.	删除联系人 （没有这个人）
# 3.	修改手机 
# 4.	查询所有用户
# 5.	根据姓名查找手机号码
# 6.     退出
#tip：考虑到一个人可能有两个手机号码，所以支持一个人两个号码,但是备注好（内容）。
#如果查找不到显示“查无此人”并且自动还回主菜单
#退出5号，可以直接提出程序，（理论上你的电话本也拜拜了。）
#上一次用列表发现非常啰嗦，这次换成字典，简洁很多


def man():            #菜单界面
    print("---------------------------------------------")
    print("请选择你要的功能")
    print("1.增加姓名和手机\n")
    print("2.删除姓名\n")
    print("3.修改手机\n")
    print("4.查询所有用户\n")
    print("5.根据姓名查找手机号\n")
    print("6.退出\n")


def only(name,a={}):   #判断字典里面是否有重名的
    if name in a.keys():
        return(1)
    else:
        return(0)



#主程序
ls={}               #通讯录
option=0          #功能选择
while option !=6:
    man()
    option=int(input())
    if option==1:          #添加姓名和电话
        print("请输入姓名\n")
        a=input()
        if only(a,ls)==0:
            print("请输入电话\n")
            b=input()
            ls[a]=b
        else:
            print("已经存在，请改成新的备注之后再添加")



    elif option==2:               #删除电话
        a=input("请输入要删除的姓名\n")
        if only(a,ls)==1:          
            ls.pop(a)
        else:
            print("查无此人")


    elif option==3:           #修改电话
        a=input("你要修改谁的手机号码\n")
        if only(a,ls)==1:
            new_iphone=int(input("请输入要修改的手机号码\n"))
            ls[a]=new_iphone
        else:
            print("查无此人")



    elif option==4:         #显示所有电话
        for j,k in ls.items():
            print("{}的电话是{}".format(j,k))
            print('\n')
        
    elif option==5:          #查找电话
        a=input("你要查找谁的手机号码\n")
        if only(a,ls)==1:
            print("{}此人的手机号码是{}".format(a,ls[a]))
        else:
            print("查无此人")