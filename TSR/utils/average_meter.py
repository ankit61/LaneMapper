class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.__sum = 0
        self.__count = 0
        self.__last = 0

    def update(self, val):
        self.__count += 1
        self.__sum += val
        self.__las = val

    def avg(self):
        return self.__sum / self.__count

    def last(self):
        return self.__last
    