import numpy as np

class NumpyDeque(object):
    def __init__(self, shape:tuple) -> None:
        self.shape_arr = shape

        self.array = np.zeros((self.shape_arr), dtype=np.float32)

    def __len__(self):
        return self.shape_arr[1]
    
    def append(self, els):
        assert els.shape[0] == self.shape_arr[0] 

        self.array = np.roll(self.array, els.shape[1], axis=1)
        self.array[:,0:els.shape[1]] = els.astype(np.float32)

    def reset(self, vecs=None):
        if vecs is None:
            self.array = np.zeros((self.shape_arr), dtype=np.float32)
        elif isinstance(vecs,np.ndarray):
            self.array[vecs==1] = 0.
            
    def __call__(self):
        return self.array
    def __repr__(self):
        return str(self.array)
    def __array__(self, dtype=None):
        if dtype:
            return self.array.astype(dtype)
        return self.array
    @property
    def shape(self):
        return self.shape_arr

if __name__=='__main__':
    test_qeue = NumpyDeque((3,5))
    print(type(test_qeue.array))
    print(test_qeue)
    ones_1 = np.ones((3,1))
    twos_2 = np.ones((3,2))*2
    test_qeue.append(ones_1)
    print(test_qeue)
    test_qeue.append(twos_2)
    print(test_qeue)
    test_qeue.reset(np.array([0,1,0]))
    print(test_qeue)

