from train import Train
from test import Test

def main():
    train_obj = Train()
    train_obj.train_mode()

    test_obj = Test()
    test_obj.test_mode()



if __name__ == '__main__':
    main()