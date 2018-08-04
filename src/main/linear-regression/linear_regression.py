"""###########################################################
###                                                        ###
##    This script models the Linear Regression algorithm    ##
###                                                        ###
###########################################################"""

from io_manager import *


def main(data_file):
    data = load(data_file)
    print(data)


# if __name__ == '__main__':
#    main('resources/data_files/ex1data1.txt')
print('Program started...')


main('data_files/ex1data1.txt')
