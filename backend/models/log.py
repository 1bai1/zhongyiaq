import json
import os
import uuid
from datetime import datetime

class Log:
    def __init__(self, id=None, user_id=None, username=None, ip=None, action=None, details=None, timestamp=None):
        self.id = id or str(uuid.uuid4())
        self.user_id = user_id
        self.username = username
        self.ip = ip
        self.action = action
        self.details = details or {}
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'username': self.username,
            'ip': self.ip,
            'action': self.action,
            'details': self.details,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data.get('id'),
            user_id=data.get('user_id'),
            username=data.get('username'),
            ip=data.get('ip'),
            action=data.get('action'),
            details=data.get('details', {}),
            timestamp=data.get('timestamp')
        )

class LogManager:
    def __init__(self, file_path='data/logs.json'):
        self.file_path = file_path
        # 确保数据目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 如果文件不存在，创建空的日志列表文件
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def get_all_logs(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                logs_data = json.load(f)
                return [Log.from_dict(log_data) for log_data in logs_data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def get_logs_by_user(self, user_id):
        logs = self.get_all_logs()
        return [log for log in logs if log.user_id == user_id]
    
    def get_login_logs(self):
        logs = self.get_all_logs()
        return [log for log in logs if log.action == 'login']
    
    def add_log(self, log):
        logs = self.get_all_logs()
        logs.append(log)
        # 按时间戳降序排序
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        # 保存到文件
        self._save_logs([log.to_dict() for log in logs])
        return log
    
    def create_log(self, user_id, username, ip, action, details=None):
        log = Log(
            user_id=user_id,
            username=username,
            ip=ip,
            action=action,
            details=details or {},
            timestamp=datetime.now().isoformat()
        )
        return self.add_log(log)
    
    def _save_logs(self, logs_data):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=4, ensure_ascii=False) 