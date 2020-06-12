# coding=utf-8

import time
import datetime
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

'''
print('自动化测试')
time.sleep(10)
print('测试完毕')
'''
'''
while True:
    if os.path.exists('/home/panyirong/File/my_code/f'):
        os.system('python reorder_table.py')
        break
    else:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('正在等待调序得分文件生成......')
        time.sleep(300)

if os.path.exists('/home/panyirong/File/my_code/test'):
    os.system('cd /home/panyirong/enworking')
    os.system('nohup nice ~/moses/scripts/training/mert-moses.pl ~/corpus/mert.true.ch ~/corpus/mert.true.uy ~/moses/bin/moses train/model/moses.ini --mertdir ~/moses/bin/  &> uy_test.mert.out &')

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('完成任务！')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
'''
my_time = datetime.datetime(2017,5,12,2,0,0)
while datetime.datetime.now() < my_time:
    time.sleep(300)

os.chdir('/home/panyirong/enmslr')
os.system('nohup nice ~/moses/scripts/training/mert-moses.pl ~/encorpus/mert.true.ch ~/encorpus/mert.true.en ~/moses/bin/moses train/model/moses.ini --mertdir ~/moses/bin/  &> mert.out &')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))