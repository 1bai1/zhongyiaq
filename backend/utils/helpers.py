import os
from flask import request
import uuid
from werkzeug.utils import secure_filename
from datetime import datetime

def get_client_ip():
    """获取客户端IP地址"""
    if 'X-Forwarded-For' in request.headers:
        # 处理代理
        ip = request.headers['X-Forwarded-For'].split(',')[0].strip()
    else:
        ip = request.remote_addr or '未知IP'
    return ip

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def generate_unique_filename(filename):
    """生成唯一的文件名"""
    # 获取文件扩展名
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    # 生成唯一的文件名
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
    if ext:
        unique_filename = f"{unique_filename}.{ext}"
    return unique_filename

def save_uploaded_file(file, upload_folder, allowed_extensions):
    """保存上传的文件并返回保存路径"""
    if file and allowed_file(file.filename, allowed_extensions):
        filename = secure_filename(file.filename)
        unique_filename = generate_unique_filename(filename)
        file_path = os.path.join(upload_folder, unique_filename)
        file.save(file_path)
        return {
            'original_filename': filename,
            'saved_filename': unique_filename,
            'file_path': file_path,
            'relative_path': os.path.join('uploads', unique_filename)
        }
    return None

def format_datetime(dt_str):
    """格式化日期时间字符串"""
    try:
        if isinstance(dt_str, str):
            dt = datetime.fromisoformat(dt_str)
        else:
            dt = dt_str
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return dt_str 