
import os


def makedir(num_name, dirRoot):
    '''
    os.makedirs(dirRoot + '/left/' + str(num_name) + '/0-20')
    os.makedirs(dirRoot + '/right/' + str(num_name) + '/0-20')
    '''
    os.makedirs(dirRoot + '/all/' + str(num_name) + '/20-25')
    os.makedirs(dirRoot + '/all/' + str(num_name) + '/25-30')
    os.makedirs(dirRoot + '/all/' + str(num_name) + '/30')

    os.makedirs(dirRoot + '/left/' + str(num_name) + '/20-25')
    os.makedirs(dirRoot + '/right/' + str(num_name) + '/20-25')
    os.makedirs(dirRoot + '/left/' + str(num_name) + '/25-30')
    os.makedirs(dirRoot + '/right/' + str(num_name) + '/25-30')
    os.makedirs(dirRoot + '/left/' + str(num_name) + '/30')
    os.makedirs(dirRoot + '/right/' + str(num_name) + '/30')