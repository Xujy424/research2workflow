'''
    2025/11/04 子类作为父类参数时，父类方法调用测试
'''


class Dog:

    def __init__(self,size,color,age, happydog):
        self.size = size
        self.color = color
        self.age = age
        self.happydog = happydog

    def run(self):
        raise NotImplementedError

    def eat(self):
        print(self.happydog.age)
        print('i m a dog i m eating!')


class HappyDog(Dog):

    def __init__(self, age):
        self.age = age

    def run(self):
        print('i m a happy dog! i m running')


class AlphaDog(Dog):
    def __init__(self, age, happydog):
        self.age = age
        self.happydog = happydog
    def run(self):
        print('i m a alpha dog! i m running')


if __name__ == '__main__':

    xjy = HappyDog(25)
    cbw = AlphaDog(25, xjy)
    cbw.run()
    cbw.eat()







