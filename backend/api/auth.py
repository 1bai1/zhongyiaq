from flask import Blueprint, render_template, redirect, url_for, request, flash, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from backend.models.user import UserManager
from backend.utils.log_manager import LogManager
from backend.utils.helpers import get_client_ip

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    # 如果用户已经登录，则重定向到首页
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 验证表单输入
        if not username or not password:
            flash('请输入用户名和密码', 'error')
            return render_template('login.html')
        
        # 获取用户
        user_manager = UserManager()
        user = user_manager.get_user_by_username(username)
        
        # 验证用户和密码
        if user is None or not user.check_password(password):
            flash('用户名或密码错误', 'error')
            return render_template('login.html')
        
        # 登录用户
        login_user(user)
        
        # 记录登录日志
        log_manager = LogManager()
        ip = get_client_ip()
        log_manager.create_log(
            user_id=user.id,
            username=user.username,
            ip=ip,
            action='login',
            details={'method': 'password'}
        )
        
        # 重定向到首页
        next_page = request.args.get('next')
        if next_page:
            return redirect(next_page)
        return redirect(url_for('main.dashboard'))
    
    return render_template('login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    # 如果用户已经登录，则重定向到首页
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # 验证表单输入
        if not username or not password:
            flash('请输入用户名和密码', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('两次输入的密码不匹配', 'error')
            return render_template('register.html')
        
        # 创建用户
        user_manager = UserManager()
        user = user_manager.create_user(username, password)
        
        if user is None:
            flash('用户名已存在', 'error')
            return render_template('register.html')
        
        # 记录注册日志
        log_manager = LogManager()
        ip = get_client_ip()
        log_manager.create_log(
            user_id=user.id,
            username=user.username,
            ip=ip,
            action='register'
        )
        
        flash('注册成功，请登录', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    # 记录退出日志
    log_manager = LogManager()
    ip = get_client_ip()
    log_manager.create_log(
        user_id=current_user.id,
        username=current_user.username,
        ip=ip,
        action='logout'
    )
    
    logout_user()
    flash('您已成功退出登录', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        # 验证表单输入
        if not current_password or not new_password or not confirm_password:
            flash('请填写所有字段', 'error')
            return render_template('change_password.html')
        
        if new_password != confirm_password:
            flash('两次输入的新密码不匹配', 'error')
            return render_template('change_password.html')
        
        # 验证当前密码
        if not current_user.check_password(current_password):
            flash('当前密码不正确', 'error')
            return render_template('change_password.html')
        
        # 更新密码
        user_manager = UserManager()
        success = user_manager.update_password(current_user.id, new_password)
        
        if success:
            # 记录密码修改日志
            log_manager = LogManager()
            ip = get_client_ip()
            log_manager.create_log(
                user_id=current_user.id,
                username=current_user.username,
                ip=ip,
                action='change_password'
            )
            
            flash('密码已成功修改', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('密码修改失败', 'error')
    
    return render_template('change_password.html') 