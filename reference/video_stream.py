# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         # ʹ��generator���������Ƶ���� ÿ�����������content������image/jpeg
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
#
# @app.route('/video_feed')  # �����ַ������Ƶ����Ӧ
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# class VideoCamera(object):
#     def __init__(self):
#         # ͨ��opencv��ȡʵʱ��Ƶ��
#         self.video = cv2.VideoCapture(0)
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, image = self.video.read()
#         # ��Ϊopencv��ȡ��ͼƬ����jpeg��ʽ�����Ҫ��motion JPEGģʽ��Ҫ�Ƚ�ͼƬת���jpg��ʽͼƬ
#         ret, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()
