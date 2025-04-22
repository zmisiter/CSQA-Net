import os
import platform
import time

import torch


class Log:  # 关于打印的基础的东西
    def __init__(self, fname=None, write=True):
        super(Log, self).__init__()
        self.write = write
        time_name = time.strftime('%Y-%m-%d_%H-%M', time.localtime())  # 和日志中保存输出结果的文件名中的时间是对应的
        if fname:
            self.fname = os.path.join(fname, time_name + '.log')  # 三个文件、log日志 markdown笔记软件、txt
            self.tname = os.path.join(fname, time_name + '.md')
            self.mname = os.path.join(fname, 'model' + '.txt')
        else:
            self.fname = time_name + '.log'
            self.tname = time_name + '.md'
            self.mname = time_name + '.txt'

        # Create the file
        if self.write:
            with open(self.fname, 'w') as f:
                pass

            with open(self.tname, 'w') as f:
                pass  # https://zhuanlan.zhihu.com/p/164598216

            with open(self.mname, 'w') as f:
                pass

    def info(self, *info, end='\n'):  # 终端输出一遍，再写入文件一遍
        print(*info, flush=True, end=end)
        if self.write:
            with open(self.fname, 'a+') as f:
                print(*info, file=f, flush=True, end=end)

    def markdown(self, *info, end='\n'):
        if self.write:
            with open(self.tname,
                      'a+') as f:  # mode=a+，可读可写, 可以不存在, https://blog.csdn.net/u011985712/article/details/79852261
                print(*info, file=f, flush=True, end=end)
        pass

    def save(self, *info):
        if self.write:
            with open(self.mname, 'w') as f:  # python用来打开本地文件的，mode='w'只写，https://www.jianshu.com/p/ce2d30f7ec26
                print(*info, file=f,
                      flush=True)
# flush=True会在print结束后，立即将内存中的东西显示到屏幕上，清空缓存 https://blog.csdn.net/u013985241/article/details/86653356


def PTitle(log, rank=0):  # 想打印出来的话，在setup_functions的SetupLogs函数里面的，想加的话可以选中注释PTitle
    if rank not in [-1, 0]: return  # 函数外面可能需要写config.local_rank，函数内部可以直接写rank
    log.info('=' * 80)
    log.info('MBVT : Multiscale Blend Vision Transformer for Fine-Grained Image Classification\n'  # 标题
             '                            Pytorch Implementation')
    log.info('=' * 80)
    log.info('Author: Zz           Institute: Anhui University           Date: 2021-12-20')  # 作者、机构、时间
    log.info('-' * 80)
    log.info(f'Python Version: {platform.python_version()}         '  # 版本
             f'Pytorch Version: {torch.__version__}         Cuda Version: {torch.version.cuda}')
    log.info('-' * 80, '\n')
    pass


class PMarkdownTable:  # markdown可以通过纯文本字符串实现表格
    def __init__(self, log, titles, rank=0):
        if rank not in [-1, 0]: return
        super(PMarkdownTable, self).__init__()
        title_line = '| '
        align_line = '| '
        for i in range(len(titles)):
            title_line = title_line + titles[i] + ' |'
            align_line = align_line + '--- |'
        log.markdown(title_line)
        log.markdown(align_line)

    def add(self, log, values, rank=0):
        if rank not in [-1, 0]: return
        value_line = '| '
        for i in range(len(values)):
            value_line = value_line + str(values[i]) + '|'
        log.markdown(value_line)

    pass


def PSetting(log, title=None, param_name=None, values=None, newline=3, rank=0):  # 可视化打印，为了美观；输出在终端的参数
    if rank not in [-1, 0]: return
    if title is not None:
        log.info('=' * 28, '{:^22}'.format(title), '=' * 28)  # 终端输出左右各28个等号，中间字符占22位，若字符不到22位则空格填充
    for i, (name, value) in enumerate(
            zip(param_name, values)):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        name = str(name)  # zip函数的作用：https://bbs.huaweicloud.com/blogs/302872
        param_name = list(param_name)  # 为了拿到param的最后一个名字
        if isinstance(value, tuple):
            value = f'{value[0]},{value[1]}'  #  加了f，''中出现了{}括起来的表达式，则会被传过来的实际数据替换
        if isinstance(value, list):
            value = str(value)  # 把列表变成字符串
        if value is None:
            value = f'None'  # 这里是不是可以不用f格式化字符串
        if newline == 3:
            if (i + 1) % newline == 0 and name != param_name[-1]:
                log.info(f'{name:14}{value :<12}')  # name占14个字符，不够用空格填充；
                log.info('- ' * 40)
            else:
                log.info(f'{name:14}{value :<12}', end='  ')

        else:  # newline==2
            if len(name) < 14:
                if (i + 1) % newline == 0 and name != param_name[-1]:
                    log.info(f'{name:14}{value :<23}')  # 对应第二列的字符格式
                    log.info('- ' * 40)
                else:
                    log.info(f'{name:14}{value :<23}', end='   ')  # name占位14，value左对齐占位23，结尾三个空格
            else:
                if (i + 1) % newline == 0 and name != param_name[-1]:
                    log.info(f'{name:18}{value :<19}')
                    log.info('- ' * 40)
                else:
                    log.info(f'{name:18}{value :<19}', end='   ')
    log.info()  # 这里是不是可以去掉？


def sub_title(log, title, rank=0):  # 副标题，拆开打印的，model structure就是在后面打印出来的
    if rank not in [-1, 0]: return
    if len(title) < 22:
        log.info('=' * 28, '{:^22}'.format(title), '=' * 28)
    elif len(title) < 30:
        log.info('=' * 24, '{:^30}'.format(title), '=' * 24)
    else:
        log.info('=' * 20, '{:^38}'.format(title), '=' * 20)
