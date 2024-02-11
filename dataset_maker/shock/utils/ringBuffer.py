import random

class ringBuffer:
    def __init__(self, buffer:list, shuffle:bool=False):
        if shuffle:
            random.shuffle(buffer)
        self.__buffer = buffer
        self.__maxSize = len(buffer)
        self.__head = 0

    def sample(self, num:int) -> list:
        sampleList = []
        for idx in range(num):
            sampleList.append(self.__buffer[self.__head])
            self.__head = (self.__head + 1) % self.__maxSize
        return sampleList

    @property
    def data(self):
        return self.__buffer


        
