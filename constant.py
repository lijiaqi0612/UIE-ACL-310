"""
-*- coding:utf-8 -*-
CREATED:  2020-8-19 10:00:00
MODIFIED:
"""
from enum import Enum
# error code
ACL_ERROR_NONE = 0

# data format
ACL_FORMAT_UNDEFINED = -1
ACL_FORMAT_NCHW = 0
ACL_FORMAT_NHWC = 1
ACL_FORMAT_ND = 2
ACL_FORMAT_NC1HWC0 = 3
ACL_FORMAT_FRACTAL_Z = 4


# rule for mem
class MallocType(Enum):
    ACL_MEM_MALLOC_HUGE_FIRST = 0   # 优先申请大页内存，如果大页内存不够，则使用普通页的内存。
    ACL_MEM_MALLOC_HUGE_ONLY = 1    #  仅申请大页，如果大页内存不够，则返回错误。
    ACL_MEM_MALLOC_NORMAL_ONLY = 2  # 仅申请普通页。


# rule for memory copy
class MemcpyType(Enum):
    ACL_MEMCPY_HOST_TO_HOST = 0
    ACL_MEMCPY_HOST_TO_DEVICE = 1
    ACL_MEMCPY_DEVICE_TO_HOST = 2
    ACL_MEMCPY_DEVICE_TO_DEVICE = 3


class NpyType(Enum):
    NPY_BOOL = 0
    NPY_BYTE = 1
    NPY_UBYTE = 2
    NPY_SHORT = 3
    NPY_USHORT = 4
    NPY_INT = 5
    NPY_UINT = 6
    NPY_LONG = 7
    NPY_ULONG = 8
    NPY_LONGLONG = 9
    NPY_ULONGLONG = 10
    NPY_FLOAT32 = 11


ACL_CALLBACK_NO_BLOCK = 0
ACL_CALLBACK_BLOCK = 1
