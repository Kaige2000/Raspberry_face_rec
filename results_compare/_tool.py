# coding: utf-8
# 工具包
# coding: utf-8
import numpy as np
from scipy.spatial.distance import pdist


def result_print(test_num, detect_error, detect_error_rate, identify_error, identify_error_rate):
    print("same face test start")
    print('the num of sample is %d' % test_num)
    print('detect error num is %d, detect error rate is %s' % (detect_error, detect_error_rate))
    print('identify error num is %d, identify error rate is %s' % (identify_error, identify_error_rate))


# def face_difference(face_encodings, face_to_compare, method):
#     if len(face_encodings) == 0:
#         return np.empty((0))
#     if method == 'e':
#         return np.linalg.norm(face_encodings - face_to_compare, axis=1)
#     elif method == 'm':
#         All_Manhattan = []
#         for face_encoding in face_encodings:
#             m_distance = np.sum(np.abs(face_encoding - face_to_compare))
#             All_Manhattan.append(m_distance)
#         return np.array(All_Manhattan)
#     elif method == 'c':
#         All_Cosine = []
#         for face_encoding in face_encodings:
#             Cosine = 1 - pdist(np.vstack([face_encoding, face_to_compare]), 'cosine')
#             All_Cosine.append(Cosine[0])
#         return np.array(All_Cosine)
#     elif method == 'p':
#         All_pccs = []
#         for face_encoding in face_encodings:
#             pccs = np.corrcoef([face_encoding, face_to_compare])[1][0]
#             All_pccs.append(pccs)
#         return np.array(All_pccs)
#
# def compare_face(known_face_encodings, face_encoding_to_check, tolerance, method):
#     if method == 'e':
#         return
#     elif method == 'm':
#         return
#     elif method == 'c':
#         return
#     elif method == 'p':
#         return


# 计算欧式距离并返回值
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# 通过比较欧式距离实现人脸匹配，返回比较列表
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=4):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


# 计算曼哈顿距离并返回值
def m_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    All_Manhattan = []
    for face_encoding in face_encodings:
        m_distance = np.sum(np.abs(face_encoding - face_to_compare))
        All_Manhattan.append(m_distance)
    return np.array(All_Manhattan)


# 通过比较曼哈顿距离实现人脸匹配，返回比较列表
def m_compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.96):
    return list(m_face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


# 计算余弦距离并返回值
def cos_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    All_Cosine = []
    for face_encoding in face_encodings:
        Cosine = 1 - pdist(np.vstack([face_encoding, face_to_compare]), 'cosine')
        All_Cosine.append(Cosine[0])
    return np.array(All_Cosine)


# 通过比较余弦距离实现人脸匹配，返回比较列表
def cos_compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.96):
    return list(cos_face_distance(known_face_encodings, face_encoding_to_check) >= tolerance)


# 计算皮尔逊系数并返回值
def p_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    All_pccs = []
    for face_encoding in face_encodings:
        pccs = np.corrcoef([face_encoding, face_to_compare])[1][0]
        All_pccs.append(pccs)
    return np.array(All_pccs)


# 通过比较皮尔逊系数实现人脸匹配，返回比较列表
def p_compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.98):
    return list(p_face_distance(known_face_encodings, face_encoding_to_check) >= tolerance)
