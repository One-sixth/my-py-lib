'''
简单数据桶
分开储存数据和索引
已储存数据不可修改
'''

import io
import struct


def _get_file_size(fp):
    fp.seek(0, io.SEEK_END)
    return fp.tell()


def _read_uint64(fp, pos):
    fp.seek(pos, io.SEEK_SET)
    buf = fp.read(8)
    assert len(buf) == 8
    return struct.unpack('<Q', buf)[0]


def _write_uint64(fp, data):
    n = fp.write(data.to_bytes(8, 'little', signed=False))
    assert n == 8


def _read_buf(fp, pos, size):
    fp.seek(pos, io.SEEK_SET)
    buf = fp.read(size)
    assert len(buf) == size
    return buf


def _write_buf(fp, data):
    n = fp.write(data)
    assert n == len(data)


class SimpleBucketReader:
    def __init__(self, file):
        '''
        :param file: 要写入的文件路径，也可以传入一个列表，指定为可写的文件句柄，[data_fp, idx_fp]
        '''
        if isinstance(file, (list, tuple)):
            self.data_file_path = None
            self.idx_file_path = None
            self.data_file = file[0]
            self.idx_file = file[1]
            assert hasattr(self.data_file, 'read') and hasattr(self.idx_file, 'read')

        else:
            self.data_file_path = file + '.data'
            self.idx_file_path = file + '.idx'
            self.data_file = open(self.data_file_path, 'rb')
            self.idx_file = open(self.idx_file_path, 'rb')

        self.data_file_size = _get_file_size(self.data_file)
        self.idx_file_size = _get_file_size(self.idx_file)

    def read(self, file_idx):
        '''
        直接返回一串字节
        :return:
        '''
        assert file_idx >= 0
        assert file_idx < self.get_file_num()

        # 读取索引
        if file_idx == 0:
            pos1 = 0
            pos2 = _read_uint64(self.idx_file, file_idx * 8)
        else:
            pos1 = _read_uint64(self.idx_file, (file_idx-1) * 8)
            pos2 = _read_uint64(self.idx_file, file_idx * 8)

        buf = _read_buf(self.data_file, pos1, pos2 - pos1)
        return buf

    def get_file_num(self):
        '''
        返回已有文件数量
        :return:
        '''
        return self.idx_file_size // 8

    def __getitem__(self, item):
        return self.read(item)

    def __len__(self):
        return self.get_file_num()


class SimpleBucketWriter:
    def __init__(self, file):
        '''
        :param file: 要写入的文件路径，也可以传入一个列表，指定为可写的文件句柄，[data_fp, idx_fp]
        '''
        if isinstance(file, (list, tuple)):
            self.data_file_path = None
            self.idx_file_path = None
            self.data_file = file[0]
            self.idx_file = file[1]
            assert hasattr(self.data_file, 'write') and hasattr(self.idx_file, 'write')

        else:
            self.data_file_path = file + '.data'
            self.idx_file_path = file + '.idx'
            self.data_file = open(self.data_file_path, 'wb')
            self.idx_file = open(self.idx_file_path, 'wb')

    def write(self, data):
        '''
        增加文件，并返回文件号
        :return:
        '''
        _write_buf(self.data_file, data)
        pos2 = self.data_file.tell()
        _write_uint64(self.idx_file, pos2)

    def close(self):
        '''
        关闭文件
        '''
        self.data_file.close()
        self.idx_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    with SimpleBucketWriter('tmp_simplebucket') as w:
        w.write(b'kdjasdnisadasiujdasdoijaio')
        w.write(b'cccasd')
        w.write(b'xzxcxz')
        w.write(b'asds')
        w.write(b'xzc')
        w.write(b'wqe112')

    r = SimpleBucketReader('tmp_simplebucket')
    print(r.get_file_num())
    for i in range(r.get_file_num()):
        print(r.read(i))
