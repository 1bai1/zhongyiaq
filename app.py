from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_login import LoginManager, login_required, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
# 强制禁用CUDA，确保所有模型都使用CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'
import json
import uuid
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import threading
import time
from flask_cors import CORS

# 导入自定义模块
from backend.models.user import User, UserManager
from backend.utils.log_manager import LogManager
from backend.api.auth import auth_bp
from backend.api.main import main_bp
from backend.api.detect import detect_bp
from backend.api.llm import llm_bp
from backend.utils.helpers import get_client_ip

# 创建必要的目录
os.makedirs('logs', exist_ok=True)
os.makedirs('frontend/static/uploads', exist_ok=True)
os.makedirs('data', exist_ok=True)

# 初始化Flask应用
app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key')
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'frontend/static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB in bytes
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# 启用CORS
CORS(app)

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化LoginManager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# 注册蓝图
app.register_blueprint(auth_bp)
app.register_blueprint(main_bp)
app.register_blueprint(detect_bp)
app.register_blueprint(llm_bp)

# 用户加载函数
@login_manager.user_loader
def load_user(user_id):
    user_manager = UserManager()
    return user_manager.get_user_by_id(user_id)

# 处理404错误
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# 全局上下文处理
@app.context_processor
def inject_user():
    return dict(user=current_user)

if __name__ == '__main__':
    # 初始化数据目录
    os.makedirs('data', exist_ok=True)
    
    # 检查用户数据文件是否存在，如果不存在则创建默认用户账户
    user_manager = UserManager()
    if not os.path.exists('data/users.json'):
        # 创建默认用户账户
        default_user = User(
            id=str(uuid.uuid4()),
            username='test',
            password_hash=generate_password_hash('test123'),
            created_at=datetime.now().isoformat()
        )
        user_manager.save_user(default_user)
        print("Created default user account: test/test123")
    
    # 初始化RAG系统
    print("正在初始化RAG系统...")
    try:
        from backend.api.llm import initialize_rag_system
        rag_success = initialize_rag_system()
        if rag_success:
            print("RAG系统初始化成功")
        else:
            print("RAG系统初始化失败，将在不使用RAG的情况下运行")
    except Exception as e:
        print(f"RAG系统初始化出错: {e}")
        print("将在不使用RAG的情况下运行")
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)