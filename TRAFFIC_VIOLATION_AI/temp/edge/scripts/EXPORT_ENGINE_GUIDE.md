<!-- Hướng dẫn sử dụng -->

Cấp quyền và chạy:

'''
    Bashcd scripts
    python export_tensorrt.py --model all
'''

Chạy export từng model:
'''
    Bashpython export_tensorrt.py --model vehicle
    python export_tensorrt.py --model traffic_light
    python export_tensorrt.py --model plate
'''
Export INT8 (tối ưu tốc độ cao nhất):

'''
    Bashpython export_tensorrt.py --model all --int8
'''

Tóm tắt ngắn gọn:

    Không chạy tự động cùng server.
    Chỉ chạy thủ công khi cần export model mới hoặc cập nhật model.
    Chạy 1 lần → xong việc, sau đó dùng file .engine để inference.  

<!-- File này sẽ xuất ra các file .engine trong thư mục edge/models/, sẵn sàng để main_edge.py sử dụng TensorRT inference. -->