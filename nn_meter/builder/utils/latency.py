import math


class Latency:
    def __init__(self, avg=0, std=0):
        if isinstance(avg, str):
            avg, std = avg.split('+-')
            self.avg = float(avg)
            self.std = float(std)
        else:
            self.avg = avg
            self.std = std

    def __str__(self):
        return f'{self.avg} +- {self.std}'

    def __add__(self, rhs):
        if isinstance(rhs, Latency):
            return Latency(self.avg + rhs.avg, math.sqrt(self.std ** 2 + rhs.std ** 2))
        else:
            return Latency(self.avg + rhs, self.std)
    
    def __radd__(self, lhs):
        return self.__add__(lhs)

    def __mul__(self, rhs):
        return Latency(self.avg * rhs, self.std * rhs)

    def __rmul__(self, lhs):
        return self.__mul__(lhs)

    def __le__(self, rhs):
        return self.avg < rhs.avg
    
    def __gt__(self, rhs):
        return self.avg > rhs.avg

    def __neg__(self):
        return Latency(-self.avg, -self.std)

    def __sub__(self, rhs):
        return self + rhs.__neg__()
