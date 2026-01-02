from flask import Blueprint, render_template, redirect, url_for, jsonify, request
from flask_login import login_required, current_user
from backend.models.user import UserManager
from backend.utils.log_manager import LogManager

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    # 获取统计信息
    user_manager = UserManager()
    log_manager = LogManager()
    
    users = user_manager.get_all_users()
    logs = log_manager.get_all_logs()
    login_logs = log_manager.get_login_logs()
    
    stats = {
        'user_count': len(users),
        'login_count': len(login_logs),
        'operation_count': len(logs),
        'detection_count': len([log for log in logs if log.action in ['detect_image', 'detect_video']])
    }
    
    return render_template('dashboard.html', stats=stats)

@main_bp.route('/profile')
@login_required
def profile():
    # 获取用户登录日志
    log_manager = LogManager()
    user_logs = log_manager.get_logs_by_user(current_user.id)
    login_logs = [log for log in user_logs if log.action == 'login']
    
    return render_template('profile.html', user=current_user, login_logs=login_logs)

@main_bp.route('/api/stats')
@login_required
def get_stats():
    # 为仪表盘提供API数据
    user_manager = UserManager()
    log_manager = LogManager()
    
    users = user_manager.get_all_users()
    logs = log_manager.get_all_logs()
    login_logs = log_manager.get_login_logs()
    
    # 按日期分组统计登录次数
    login_dates = {}
    for log in login_logs:
        date = log.timestamp.split('T')[0]
        login_dates[date] = login_dates.get(date, 0) + 1
    
    # 计算检测操作数量
    detection_count = len([log for log in logs if log.action in ['detect_image', 'detect_video']])
    
    # 获取检测历史（最近的检测记录）
    detection_logs = [log for log in logs if log.action in ['detect_image', 'detect_video']]
    detection_logs.sort(key=lambda x: x.timestamp, reverse=True)
    recent_detections = detection_logs[:10]  # 最近10条
    
    # 统计行为分布
    behavior_stats = {}
    for log in detection_logs:
        if hasattr(log, 'details') and log.details and 'detections' in log.details:
            for detection in log.details['detections']:
                class_name = detection.get('class_name', '未知')
                behavior_stats[class_name] = behavior_stats.get(class_name, 0) + 1
    
    # 按日期统计检测次数
    detection_dates = {}
    for log in detection_logs:
        date = log.timestamp.split('T')[0]
        detection_dates[date] = detection_dates.get(date, 0) + 1
    
    stats = {
        'user_count': len(users),
        'login_count': len(login_logs),
        'operation_count': len(logs),
        'detection_count': detection_count,
        'login_trend': [{'date': date, 'count': count} for date, count in login_dates.items()],
        'detection_trend': [{'date': date, 'count': count} for date, count in detection_dates.items()],
        'behavior_stats': behavior_stats,
        'recent_detections': [
            {
                'id': log.id,
                'timestamp': log.timestamp,
                'action': log.action,
                'username': log.username,
                'details': log.details if hasattr(log, 'details') else {}
            } for log in recent_detections
        ]
    }
    
    return jsonify(stats)

@main_bp.route('/api/user/theme', methods=['POST'])
@login_required
def update_user_theme():
    """更新用户主题偏好"""
    try:
        data = request.get_json()
        theme = data.get('theme', 'default')
        
        # 验证主题是否有效
        valid_themes = ['default', 'tech-green', 'vibrant-orange', 'elegant-purple', 
                       'midnight-blue', 'rose-red', 'dark']
        if theme not in valid_themes:
            return jsonify({'success': False, 'error': '无效的主题'}), 400
        
        # 更新用户主题
        user_manager = UserManager()
        if user_manager.update_theme(current_user.id, theme):
            # 记录日志
            log_manager = LogManager()
            log_manager.create_log(
                user_id=current_user.id,
                username=current_user.username,
                ip='127.0.0.1',
                action='update_theme',
                details={'theme': theme}
            )
            
            return jsonify({'success': True, 'message': '主题已更新', 'theme': theme})
        else:
            return jsonify({'success': False, 'error': '更新失败'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/api/logs/detection')
@login_required
def get_detection_logs():
    """获取检测日志"""
    try:
        log_manager = LogManager()
        logs = log_manager.get_all_logs()
        
        # 过滤出检测相关的日志
        detection_logs = [log for log in logs if log.action in ['detect_image', 'detect_video', 'camera_detect']]
        
        # 按时间排序
        detection_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        # 转换为字典格式
        logs_data = []
        for log in detection_logs:
            log_dict = log.to_dict()
            logs_data.append(log_dict)
        
        return jsonify({
            'success': True,
            'logs': logs_data,
            'total': len(logs_data)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500