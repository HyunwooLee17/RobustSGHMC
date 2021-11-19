import copy

class A:
    def __init__(self,a):
        self.a=a
    def print(self):
        print(self.a,self.b,self.c)

class B:
    def __init__(self,a):
        self.a=a
    def print(self):
        print(self.a,self.b,self.c)

aa=B(A(1))
aa.a.buf=3
bb=copy.deepcopy(aa)
print(bb.a.buf)

