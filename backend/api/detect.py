import os
import time
import cv2
import numpy as np
import torch
import json
import sys
import threading
from flask import Blueprint, render_template, request, jsonify, current_app, url_for, send_from_directory, Response
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

# 导入utils模块
from backend.utils.helpers import get_client_ip, save_uploaded_file
from backend.utils.log_manager import LogManager
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import base64

# 设置最大请求大小为500MB
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB in bytes

# 创建蓝图
detect_bp = Blueprint('detect', __name__)

# 确保全局变量在模块加载时就存在
yolo_model = None
model_available = True

# 设置蓝图级别的最大请求大小
detect_bp.config = {
    'MAX_CONTENT_LENGTH': MAX_CONTENT_LENGTH
}

# 尝试导入检测模型
try:
    from ultralytics import YOLO
    model_available = True
    print("成功导入YOLO模型")
except ImportError as e:
    try:
        # 尝试从本地目录导入
        import ultralytics
        from ultralytics import YOLO
        model_available = True
        print("成功从本地目录导入YOLO模型")
    except ImportError as e:
        model_available = False
        print(f"导入YOLO模型失败: {str(e)}")
        print(f"Python路径: {sys.path}")


# 添加全局变量用于存储最新的检测状态
camera_detection_status = {
    'fps': 0,
    'detection_count': 0,
    'latest_detections': [],
    'last_update': 0
}

# 使用PIL在图像上添加中文
def add_chinese_text(img, text, position, text_color=(255, 255, 255), text_size=20):
    """在图像上添加文本，尝试使用中文字体，如果失败则回退到英文显示"""
    try:
        # 尝试使用PIL添加中文文本
        from PIL import Image, ImageDraw, ImageFont
        
        # OpenCV图像是BGR格式，需要转换为RGB
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试使用可能的字体路径
        font = None
        try:
            # 尝试使用系统字体
            font_path = None
            system_fonts = [
                "SimHei.ttf",  # 当前目录
                "static/fonts/SimHei.ttf",  # 静态目录
                "C:/Windows/Fonts/SimHei.ttf",  # Windows 系统字体
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
                "/System/Library/Fonts/STHeiti Light.ttc"  # macOS
            ]
            
            for path in system_fonts:
                if os.path.exists(path):
                    font_path = path
                    break
            
            # 加载字体
            if font_path:
                font = ImageFont.truetype(font_path, text_size)
            else:
                # 如果找不到任何中文字体，使用默认字体
                font = ImageFont.load_default()
        except Exception as e:
            print(f"加载字体出错: {str(e)}")
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=text_color)
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # 如果PIL出错，回退到OpenCV的英文文本绘制
        print(f"使用PIL添加文本失败，回退到OpenCV: {str(e)}")
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        return img

def get_available_gpu():
    """获取可用的GPU设备"""
    if not torch.cuda.is_available():
        return 'cpu'
    
    # 检查所有GPU的内存使用情况
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU设备")
    
    min_memory_used = float('inf')
    best_gpu = 0
    
    for i in range(gpu_count):
        try:
            # 获取GPU内存信息
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            used_memory = memory_allocated + memory_reserved
            
            print(f"GPU {i}: 已用内存 {used_memory / 1024**3:.2f}GB / 总内存 {total_memory / 1024**3:.2f}GB")
            
            # 选择内存使用最少的GPU
            if used_memory < min_memory_used:
                min_memory_used = used_memory
                best_gpu = i
                
        except Exception as e:
            print(f"检查GPU {i} 时出错: {e}")
            continue
    
    print(f"选择GPU {best_gpu} 进行推理 (内存使用最少: {min_memory_used / 1024**3:.2f}GB)")
    return best_gpu

def safe_model_inference(model, input_data):
    """安全的模型推理，带CUDA内存管理和强制CPU模式"""
    try:
        # 先清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制使用CPU进行推理
        try:
            # 设置环境变量强制使用CPU
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            
            # 尝试直接CPU推理
            results = model.predict(input_data, device='cpu', verbose=False)
            return results
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as cuda_error:
            if 'out of memory' in str(cuda_error).lower() or 'cuda' in str(cuda_error).lower():
                print(f"CUDA错误，强制使用CPU: {str(cuda_error)}")
                
                # 清理所有CUDA资源
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # 重新初始化模型为CPU模式
                try:
                    from ultralytics import YOLO
                    model_path = 'backend/models/best.pt'
                    cpu_model = YOLO(model_path)
                    # 强制设置为CPU
                    cpu_model.model = cpu_model.model.cpu()
                    results = cpu_model.predict(input_data, device='cpu', verbose=False)
                    return results
                except Exception as fallback_error:
                    print(f"CPU回退也失败: {str(fallback_error)}")
                    raise fallback_error
            else:
                raise cuda_error
                
    except Exception as e:
        print(f"模型推理出错: {str(e)}")
        raise e

def initialize_yolo_model():
    """初始化YOLO模型，强制使用CPU模式避免CUDA问题"""
    global yolo_model
    try:
        # 强制设置环境变量禁用CUDA
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # 清理所有CUDA资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("已清理CUDA缓存")
        
        print("强制使用CPU模式初始化模型")
        
        # 尝试加载已有模型
        model_path = 'backend/models/best.pt'  # 使用中草药识别模型
        if os.path.exists(model_path):
            print(f"正在加载模型: {model_path}")
            
            try:
                # 使用更安全的方式初始化YOLO模型
                with torch.no_grad():
                    yolo_model = YOLO(model_path)
                    
                    # 确保模型在CPU上
                    if hasattr(yolo_model, 'model') and yolo_model.model is not None:
                        yolo_model.model = yolo_model.model.cpu()
                        # 设置模型为评估模式
                        yolo_model.model.eval()
                
                # 最后清理CUDA资源
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print("模型已成功加载到CPU")
                return True
                
            except Exception as load_error:
                print(f"加载模型失败: {str(load_error)}")
                print(f"错误类型: {type(load_error).__name__}")
                
                # 尝试清理并重试
                if 'yolo_model' in globals() and yolo_model is not None:
                    try:
                        del yolo_model
                        yolo_model = None
                    except:
                        pass
                        
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                return False
                
        else:
            # 无法找到模型文件
            print(f"无法找到模型文件: {model_path}")
            return False
            
    except Exception as e:
        print(f"初始化YOLO模型时出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        return False

@detect_bp.route('/detect')
@login_required
def detect_page():
    """显示检测页面"""
    global model_available
    return render_template('detect.html', model_available=model_available)

@detect_bp.route('/api/detect/status', methods=['GET'])
@login_required
def get_model_status():
    """获取模型状态"""
    global yolo_model, model_available
    
    # 如果模型未加载，尝试加载
    if yolo_model is None and model_available:
        initialize_yolo_model()
    
    status = {
        'available': model_available and yolo_model is not None,
        'model_loaded': yolo_model is not None,
        'model_path': 'backend/models/best.pt' if yolo_model is not None else None
    }
    
    return jsonify(status)

@detect_bp.route('/api/detect/image', methods=['POST', 'HEAD'])
@login_required
def detect_image():
    """处理图像检测请求"""
    global yolo_model
    
    try:
        # 确保模型已加载
        if yolo_model is None:
            print("模型未加载，尝试加载模型...")
            if not initialize_yolo_model():
                return jsonify({'error': '模型未加载或不可用'}), 500
        
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400
        
        # 保存上传的图像
        file_info = save_uploaded_file(file, current_app.config['UPLOAD_FOLDER'], current_app.config['ALLOWED_EXTENSIONS'])
        if not file_info:
            return jsonify({'error': '不支持的文件类型'}), 400
        
        print(f"开始处理图像: {file_info['file_path']}")
        
        # 进行检测
        start_time = time.time()
        results = safe_model_inference(yolo_model, file_info['file_path'])
        elapsed_time = time.time() - start_time
        
        print(f"检测完成，用时: {elapsed_time:.2f}秒")
        
        # 保存检测结果图像
        result_img = results[0].plot()
        result_filename = f"result_{file_info['saved_filename']}"
        result_path = os.path.join(current_app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        # 获取检测结果
        detection_results = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                if hasattr(box, 'cls') and hasattr(result, 'names'):
                    class_id = int(box.cls[0]) if isinstance(box.cls, np.ndarray) else int(box.cls)
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0]) if isinstance(box.conf, np.ndarray) else float(box.conf)
                    xyxy = box.xyxy[0].tolist() if isinstance(box.xyxy, np.ndarray) else box.xyxy.tolist()
                    
                    detection_results.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'box': xyxy
                    })
        
        # 记录检测日志
        try:
            log_manager = LogManager()
            
            # 准备详细日志信息
            log_details = {
                'filename': file_info['original_filename'],
                'detection_count': len(detection_results),
                'elapsed_time': elapsed_time,
                'detections': [
                    {
                        'class_id': item['class_id'],
                        'class_name': item['class_name'],
                        'confidence': item['confidence'],
                        'box': item['box']
                    } for item in detection_results
                ],
                'model_used': 'YOLOv11 - backend/models/best.pt',
                'image_size': [results[0].orig_shape[1], results[0].orig_shape[0]],
                'result_filename': result_filename
            }
            
            # 打印调试信息
            print("=== 创建检测日志 ===")
            print(f"用户ID: {current_user.id}")
            print(f"用户名: {current_user.username}")
            print(f"IP: {get_client_ip()}")
            print(f"操作: detect_image")
            print(f"检测对象数量: {len(detection_results)}")
            
            # 记录日志
            log_entry = log_manager.create_log(
                user_id=current_user.id,
                username=current_user.username,
                ip=get_client_ip(),
                action='detect_image',
                details=log_details
            )
            
            print(f"日志记录完成: {log_entry}")
        except Exception as e:
            print(f"记录日志时出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        # 确保URL路径使用正斜杠
        original_image_path = file_info['relative_path'].replace('\\', '/')
        result_image_path = f"uploads/{result_filename}".replace('\\', '/')
        
        return jsonify({
            'success': True,
            'message': f'检测完成，用时: {elapsed_time:.2f}秒',
            'original_image': url_for('static', filename=original_image_path),
            'result_image': url_for('static', filename=result_image_path),
            'detections': detection_results,
            'elapsed_time': elapsed_time
        })
    
    except Exception as e:
        print(f"检测过程出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'检测过程出错: {str(e)}'}), 500

@detect_bp.route('/api/detect/video', methods=['POST'])
@login_required
def detect_video():
    """处理视频上传请求"""
    print("收到视频上传请求")  # 添加调试日志
    
    # 检查是否有文件上传
    if 'file' not in request.files:
        print("没有上传文件")  # 添加调试日志
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("未选择文件")  # 添加调试日志
        return jsonify({'error': '未选择文件'}), 400
    
    print(f"接收到文件: {file.filename}")  # 添加调试日志
    
    # 保存上传的视频
    file_info = save_uploaded_file(file, current_app.config['UPLOAD_FOLDER'], current_app.config['ALLOWED_EXTENSIONS'])
    if not file_info:
        print("不支持的文件类型")  # 添加调试日志
        return jsonify({'error': '不支持的文件类型'}), 400
    
    print(f"文件已保存: {file_info['file_path']}")  # 添加调试日志
    
    # 创建任务ID
    task_id = f"video_{int(time.time())}"
    print(f"创建任务ID: {task_id}")  # 添加调试日志
    
    # 保存当前用户信息
    user_id = current_user.id
    username = current_user.username
    
    # 创建任务状态文件
    task_data = {
        'status': 'uploaded',  # 初始状态为已上传
        'progress': 0,
        'file_info': file_info,
        'result_path': None,
        'error': None,
        'total_frames': 0,
        'processed_frames': 0,
        'current_frame': None
    }
    
    # 确保data目录存在
    os.makedirs('data', exist_ok=True)
    task_file = os.path.join('data', f"{task_id}.json")
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(task_data, f)
    
    print(f"任务状态文件已创建: {task_file}")  # 添加调试日志
    
    # 启动异步处理线程
    try:
        # 创建应用上下文
        app = current_app._get_current_object()
        
        def process_with_context():
            with app.app_context():
                process_video(task_id, file_info, user_id, username)
        
        thread = threading.Thread(target=process_with_context)
        thread.daemon = True  # 设置为守护线程
        thread.start()
        print(f"处理线程已启动: {thread.ident}")  # 添加调试日志
        
        # 等待线程启动
        time.sleep(0.1)
        
        # 检查线程是否还在运行
        if not thread.is_alive():
            raise Exception("处理线程启动失败")
        
    except Exception as e:
        print(f"启动处理线程失败: {str(e)}")  # 添加调试日志
        import traceback
        print(traceback.format_exc())
        
        # 更新任务状态为错误
        task_data['status'] = 'error'
        task_data['error'] = f'启动处理线程失败: {str(e)}'
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f)
        
        return jsonify({'error': f'启动处理线程失败: {str(e)}'}), 500
    
    return jsonify({
        'success': True,
        'message': '视频上传成功，开始处理',
        'task_id': task_id
    })

@detect_bp.route('/api/detect/video/status/<task_id>', methods=['GET'])
@login_required
def get_video_status(task_id):
    """获取视频处理状态"""
    task_file = os.path.join('data', f"{task_id}.json")
    
    if not os.path.exists(task_file):
        return jsonify({'error': '任务不存在'}), 404
    
    with open(task_file, 'r', encoding='utf-8') as f:
        task_data = json.load(f)
    
    if task_data['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': task_data['error']
        }), 500
    
    if task_data['status'] == 'completed':
        return jsonify({
            'status': 'completed',
            'result_path': task_data['result_path'],
            'elapsed_time': task_data.get('elapsed_time', 0)
        })
    
    # 返回处理进度
    return jsonify({
        'status': task_data['status'],
        'progress': task_data['progress'],
        'processed_frames': task_data['processed_frames'],
        'total_frames': task_data['total_frames'],
        'current_frame': task_data['current_frame']
    })

def process_video(task_id, file_info, user_id, username):
    """异步处理视频检测"""
    global yolo_model
    task_file = os.path.join('data', f"{task_id}.json")
    
    print(f"开始处理视频任务 {task_id}")  # 添加调试日志
    
    # 创建应用上下文
    from flask import current_app
    app = current_app._get_current_object()
    
    with app.app_context():
        try:
            # 更新状态为处理中
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            task_data['status'] = 'processing'
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f)
            
            # 确保模型已加载
            if yolo_model is None:
                print("模型未加载，尝试加载模型...")
                if not initialize_yolo_model():
                    raise Exception("无法加载YOLO模型")
            
            # 检查文件是否存在
            if not os.path.exists(file_info['file_path']):
                raise Exception(f"视频文件不存在: {file_info['file_path']}")
            
            print(f"打开视频文件: {file_info['file_path']}")  # 添加调试日志
            
            # 读取视频
            video = cv2.VideoCapture(file_info['file_path'])
            if not video.isOpened():
                raise Exception("无法打开视频文件")
            
            fps = video.get(cv2.CAP_PROP_FPS)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"开始处理视频: {file_info['original_filename']}")
            print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
            
            # 更新总帧数
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            task_data['total_frames'] = total_frames
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f)
            
            # 确保输出目录存在
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            print(f"确保输出目录存在: {upload_folder}")
            
            # 创建输出视频
            result_filename = f"result_{file_info['saved_filename']}"
            result_path = os.path.join(upload_folder, result_filename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            
            if not output_video.isOpened():
                raise Exception("无法创建输出视频文件")
            
            frame_count = 0
            start_time = time.time()
            
            print("开始处理视频帧...")  # 添加调试日志
            
            while True:
                ret, frame = video.read()
                if not ret:
                    print("视频读取完成")  # 添加调试日志
                    break
                
                try:
                    print(f"处理第 {frame_count + 1} 帧")  # 添加调试日志
                    
                    # 进行检测
                    results = safe_model_inference(yolo_model, frame)
                    result_frame = results[0].plot()
                    
                    # 将当前帧转换为base64
                    _, buffer = cv2.imencode('.jpg', result_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # 写入输出视频
                    output_video.write(result_frame)
                    
                    # 更新进度
                    frame_count += 1
                    progress = min(99, int((frame_count / total_frames) * 100))
                    
                    # 更新任务状态
                    with open(task_file, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                    
                    task_data['progress'] = progress
                    task_data['processed_frames'] = frame_count
                    task_data['current_frame'] = frame_base64
                    
                    with open(task_file, 'w', encoding='utf-8') as f:
                        json.dump(task_data, f)
                    
                    print(f"已处理 {frame_count}/{total_frames} 帧，进度: {progress}%")  # 添加调试日志
                    
                    # 控制处理速度
                    time.sleep(0.03)  # 约30fps
                    
                except Exception as e:
                    print(f"处理帧 {frame_count} 时出错: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    continue
            
            # 释放资源
            video.release()
            output_video.release()
            
            elapsed_time = time.time() - start_time
            print(f"视频处理完成，用时: {elapsed_time:.2f}秒")
            
            # 更新任务完成状态
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            task_data['status'] = 'completed'
            task_data['progress'] = 100
            task_data['result_path'] = f"uploads/{result_filename}"
            task_data['elapsed_time'] = elapsed_time
            task_data['current_frame'] = None  # 清除最后一帧
            
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f)
            
            print(f"任务状态已更新为完成，结果路径: {task_data['result_path']}")
            
            # 更新检测日志
            log_manager = LogManager()
            log_details = {
                'filename': file_info['original_filename'],
                'video_info': {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'total_frames': total_frames
                },
                'elapsed_time': elapsed_time,
                'result_path': task_data['result_path']
            }
            
            # 使用本地IP地址
            client_ip = '127.0.0.1'
            
            log_entry = log_manager.create_log(
                user_id=user_id,
                username=username,
                ip=client_ip,
                action='detect_video',
                details=log_details
            )
            
            print(f"日志记录完成: {log_entry}")
            
        except Exception as e:
            print(f"视频处理出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # 更新错误状态
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            task_data['status'] = 'error'
            task_data['error'] = str(e)
            
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_data, f)

@detect_bp.route('/api/debug/logs')
@login_required
def debug_logs():
    """调试日志API"""
    log_dir = os.path.join(os.getcwd(), 'logs')
    result = {
        'success': True,
        'logs_directory': log_dir,
        'directory_exists': os.path.exists(log_dir),
        'files': [],
        'content_sample': None,
        'current_user': {
            'id': str(current_user.id),
            'username': current_user.username,
            'is_authenticated': current_user.is_authenticated
        }
    }
    
    # 检查日志目录
    if result['directory_exists']:
        # 列出所有日志文件
        result['files'] = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        
        # 检查今日日志
        today_log = f"log_{datetime.now().strftime('%Y%m%d')}.json"
        today_log_path = os.path.join(log_dir, today_log)
        
        result['today_log'] = {
            'filename': today_log,
            'exists': os.path.exists(today_log_path),
            'path': today_log_path
        }
        
        # 如果今日日志存在，读取内容
        if result['today_log']['exists']:
            try:
                with open(today_log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
                    result['today_log']['entry_count'] = len(logs)
                    # 提供最新的几条日志作为样本
                    result['today_log']['recent_entries'] = logs[-5:] if len(logs) > 0 else []
            except Exception as e:
                result['today_log']['error'] = str(e)
    
    # 测试创建日志
    try:
        log_manager = LogManager()
        test_log = log_manager.create_log(
            user_id=current_user.id,
            username=current_user.username,
            ip=get_client_ip(),
            action='debug_logs',
            details={'test': True, 'time': datetime.now().isoformat()}
        )
        result['test_log_created'] = True
        result['test_log'] = test_log
    except Exception as e:
        result['test_log_created'] = False
        result['test_log_error'] = str(e)
    
    return jsonify(result)

@detect_bp.route('/api/debug/data_logs')
@login_required
def debug_data_logs():
    """调试data/logs.json文件"""
    log_file = os.path.join('data', 'logs.json')
    result = {
        'success': True,
        'log_file': log_file,
        'file_exists': os.path.exists(log_file),
        'entries': []
    }
    
    # 检查data目录
    data_dir = 'data'
    result['data_directory'] = {
        'path': data_dir,
        'exists': os.path.exists(data_dir),
        'files': []
    }
    
    if result['data_directory']['exists']:
        result['data_directory']['files'] = os.listdir(data_dir)
    
    # 如果日志文件存在，读取内容
    if result['file_exists']:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
                result['entry_count'] = len(logs)
                # 获取检测日志数量
                detection_logs = [log for log in logs if log.get('action') == 'detect_image']
                result['detection_log_count'] = len(detection_logs)
                # 提供最新的几条检测日志作为样本
                result['recent_detection_logs'] = detection_logs[-3:] if detection_logs else []
                # 提供最新的几条日志作为样本
                result['recent_entries'] = logs[-5:] if logs else []
        except Exception as e:
            result['error'] = str(e)
    
    return jsonify(result)

@detect_bp.route('/detection_history')
@login_required
def detection_history():
    """显示检测历史记录页面"""
    return render_template('detection_history.html')

@detect_bp.route('/runs/<path:filename>')
def serve_runs(filename):
    """提供runs目录下的文件"""
    return send_from_directory('runs', filename)

@detect_bp.route('/camera_detect')
@login_required
def camera_detect():
    """显示摄像头实时检测页面"""
    global model_available
    return render_template('camera_detect.html', model_available=model_available)

@detect_bp.route('/api/detect/camera/frame', methods=['POST'])
@login_required
def detect_camera_frame():
    """处理前端发送的摄像头帧并返回检测结果"""
    global yolo_model, camera_detection_status
    
    try:
        # 确保模型已加载
        if yolo_model is None:
            if not initialize_yolo_model():
                return jsonify({'error': '模型未加载'}), 500
        
        # 获取base64编码的图像
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({'error': '未收到图像数据'}), 400
        
        # 解码base64图像
        frame_data = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        frame_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': '无法解码图像'}), 400
        
        # 执行检测
        start_time = time.time()
        results = safe_model_inference(yolo_model, frame)
        elapsed_time = time.time() - start_time
        
        # 分析检测结果
        detection_results = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if hasattr(box, 'cls') and hasattr(result, 'names'):
                    class_id = int(box.cls[0]) if isinstance(box.cls, np.ndarray) else int(box.cls)
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0]) if isinstance(box.conf, np.ndarray) else float(box.conf)
                    xyxy = box.xyxy[0].tolist() if isinstance(box.xyxy, np.ndarray) else box.xyxy.tolist()
                    
                    detection_results.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'box': xyxy
                    })
        
        # 更新全局状态
        current_time = time.time()
        camera_detection_status['detection_count'] = len(detection_results)
        camera_detection_status['latest_detections'] = detection_results
        camera_detection_status['last_update'] = current_time
        
        # 绘制检测结果
        result_frame = results[0].plot()
        
        # 编码为base64
        _, buffer = cv2.imencode('.jpg', result_frame)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'frame': f'data:image/jpeg;base64,{result_base64}',
            'detections': detection_results,
            'elapsed_time': elapsed_time
        })
    
    except Exception as e:
        print(f"处理摄像头帧时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 已移除 generate_frames 函数，改用前端发送帧的方式

@detect_bp.route('/api/detect/camera/status')
@login_required
def camera_detection_status_api():
    """获取摄像头检测状态"""
    global camera_detection_status
    
    # 创建一个新的字典，避免引用问题
    status = {
        'success': True,
        'fps': camera_detection_status.get('fps', 0),
        'detection_count': camera_detection_status.get('detection_count', 0),
        'last_update': camera_detection_status.get('last_update', 0)
    }
    
    # 只有当最后更新时间是在5秒内才返回latest_detections
    current_time = time.time()
    if current_time - camera_detection_status.get('last_update', 0) < 5:
        status['latest_detections'] = camera_detection_status.get('latest_detections', [])
    else:
        status['latest_detections'] = []
    
    return jsonify(status)

@detect_bp.route('/api/detect/download_video/<filename>')
@login_required
def download_video(filename):
    """下载检测结果视频"""
    try:
        # 安全检查：确保文件名不包含路径遍历字符
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'error': '非法文件名'}), 400
        
        # 检查文件是否存在
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 404
        
        # 检查文件是否为视频文件
        if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return jsonify({'error': '不是视频文件'}), 400
        
        # 返回文件下载
        return send_from_directory(
            current_app.config['UPLOAD_FOLDER'], 
            filename, 
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"下载视频文件时出错: {str(e)}")
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@detect_bp.route('/api/detect/save_to_history', methods=['POST'])
@login_required
def save_to_history():
    """保存检测结果到历史记录"""
    try:
        data = request.get_json()
        detection_type = data.get('type')
        
        # 准备日志信息
        log_details = {
            'type': detection_type,
            'timestamp': datetime.now().isoformat(),
            'user_id': current_user.id,
            'username': current_user.username
        }
        
        if detection_type == 'image':
            log_details.update({
                'original_image': data.get('original_image'),
                'result_image': data.get('result_image'),
                'detections': data.get('detections'),
                'elapsed_time': data.get('elapsed_time')
            })
        elif detection_type == 'video':
            log_details.update({
                'result_url': data.get('result_url'),
                'task_id': data.get('task_id')
            })
        
        # 记录日志
        log_manager = LogManager()
        log_entry = log_manager.create_log(
            user_id=current_user.id,
            username=current_user.username,
            ip=get_client_ip(),
            action='save_to_history',
            details=log_details
        )
        
        return jsonify({
            'success': True,
            'message': '已成功保存到历史记录',
            'log_id': str(log_entry)
        })
    
    except Exception as e:
        print(f"保存历史记录时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'保存历史记录时出错: {str(e)}'
        }), 500 