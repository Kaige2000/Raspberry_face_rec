# coding: utf-8
# 工具包


def result_print(test_num, detect_error, detect_error_rate, identify_error, identify_error_rate):
    print("same face test start")
    print('the num of sample is %d' % test_num)
    print('detect error num is %d, detect error rate is %s' % (detect_error, detect_error_rate))
    print('identify error num is %d, identify error rate is %s' % (identify_error, identify_error_rate))