# if seta in setb:print('true')
from multiprocessing import Pool
if __name__ == '__main__':
    pool = Pool(processes=8)
    seta = { 1,2,3,4,5,None}
    setb = { 1,2,3,4,5,6}
    print(len(seta))

    if len(setb.intersection(seta)) > 0 :
        print(True)
