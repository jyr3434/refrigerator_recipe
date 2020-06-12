import pandas as pd

# try-except 문
def safe_pop_print(list, index):
    try:
        print(list.pop(index))
    except IndexError:
        print('{} index의 값을 가져올 수 없습니다.'.format(index))

safe_pop_print([1,2,3], 5) # 5 index의 값을 가져올 수 없습니다.

# if 문
def safe_pop_print(list, index):
    if index < len(list):
        print(list.pop(index))
    else:
        print('{} index의 값을 가져올 수 없습니다.'.format(index))

safe_pop_print([1,2,3], 5) # 5 index의 값을 가져올 수 없습니다.