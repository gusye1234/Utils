from .pretty import *
from .tools import *

def process(n):
        if n < 0:
            raise ValueError(f"n should be larger than zero, but get {n}")
        if n == 1 or n== 0:
            return 1
        else:
            return process(n-1) + process(n-2)
    
def process_list(l):
    for n in l:
        process(n)

def test_multi():
    
    data = list(range(20,40))
    
    first_1, first_2, first_3, first_4 = data[::4], data[1::4], data[2::4], data[3::4]
    second = [[i] for i in data]
    
    print(first_1, first_2)
    with timer():
        goMulti(process_list, [first_1, first_2, first_3, first_4], cores=4)
    print(f"cost {timer.get()}")
    with timer():
        goMulti(process_list, second, cores=4)
    print(f"cost {timer.get()}")
    
        
        
if __name__ == "__main__":
    test_multi()