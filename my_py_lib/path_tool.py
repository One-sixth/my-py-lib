'''
路径工具
为了处理麻烦的路径名称问题
'''
import os
import sys


def split_file_path(p, base_dir=None):
    '''
    分离完整路径为 文件夹路径，文件基本名，文件后缀名
    如果 base_dir 被指定，那么将分离为 基础文件夹路径，中间文件夹路径，文件基本名，文件后缀名

    例子：
    o = split_file_path('C:/dir/123.txt')
    print(o)
    ('C:/dir', '123', '.txt')

    o = split_file_path('C:/dir/123.txt', 'C:')
    print(o)
    ('C:', '/dir', '123', '.txt')

    :param p:
    :return:
    '''
    dir_path: str
    dir_path, name = os.path.split(p)
    basename, extname = os.path.splitext(name)

    if base_dir is None:
        return dir_path, basename, extname

    assert dir_path.startswith(base_dir), f'Error! dir_path:{dir_path} must be startwith base_dir:{base_dir}'
    dir_path = dir_path.removeprefix(base_dir)
    return base_dir, dir_path, basename, extname


def insert_basename_end(p, s):
    '''
    在基本名后面插入字符串

    例子：
    s = insert_basename_end('C:/12.txt', '_a')
    print(s)
    C:/12_a.txt

    :param p: 原始文件名
    :param s: 要附加的字符串
    :return:
    '''
    dir_path, basename, extname = split_file_path(p)
    o = dir_path + basename + s + extname
    return o


def replace_extname(p, s):
    '''
    替换掉后缀名

    例子：
    s = replace_extname('C:/12.txt', '.jpg')
    print(s)
    C:/12.jpg

    :param p: 原始文件名
    :param s: 要附加的字符串
    :return:
    '''
    dir_path, basename, extname = split_file_path(p)
    o = dir_path + '/' + basename + s
    return o


def get_home_dir():
    '''
    获得当前用户家目录，支持windows，linux和macosx
    :return:
    '''
    if sys.platform == 'win32':
        homedir = os.environ['USERPROFILE']
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        homedir = os.environ['HOME']
    else:
        raise NotImplemented(f'Error! Not this system. {sys.platform}')
    return homedir
