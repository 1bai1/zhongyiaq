import json
import os
import uuid
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    def __init__(self, id, username, password_hash, created_at=None, theme='default'):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.created_at = created_at or datetime.now().isoformat()
        self.theme = theme  # 用户主题偏好
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'password_hash': self.password_hash,
            'created_at': self.created_at,
            'theme': self.theme
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data['id'],
            username=data['username'],
            password_hash=data['password_hash'],
            created_at=data.get('created_at'),
            theme=data.get('theme', 'default')
        )

class UserManager:
    def __init__(self, file_path='data/users.json'):
        self.file_path = file_path
        # 确保数据目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 如果文件不存在，创建空的用户列表文件
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def get_all_users(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
                return [User.from_dict(user_data) for user_data in users_data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def get_user_by_id(self, user_id):
        users = self.get_all_users()
        for user in users:
            if user.id == user_id:
                return user
        return None
    
    def get_user_by_username(self, username):
        users = self.get_all_users()
        for user in users:
            if user.username == username:
                return user
        return None
    
    def save_user(self, user):
        users = self.get_all_users()
        # 查找是否已存在该用户
        exists = False
        for i, existing_user in enumerate(users):
            if existing_user.id == user.id:
                users[i] = user
                exists = True
                break
        
        # 如果不存在则添加
        if not exists:
            users.append(user)
        
        # 保存到文件
        self._save_users([u.to_dict() for u in users])
        return user
    
    def create_user(self, username, password):
        # 检查用户名是否已存在
        if self.get_user_by_username(username):
            return None
        
        # 创建新用户
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            password_hash=generate_password_hash(password),
            created_at=datetime.now().isoformat()
        )
        
        # 保存用户
        self.save_user(user)
        return user
    
    def update_password(self, user_id, new_password):
        user = self.get_user_by_id(user_id)
        if user:
            user.password_hash = generate_password_hash(new_password)
            self.save_user(user)
            return True
        return False
    
    def update_theme(self, user_id, theme):
        """更新用户主题偏好"""
        user = self.get_user_by_id(user_id)
        if user:
            user.theme = theme
            self.save_user(user)
            return True
        return False
    
    def delete_user(self, user_id):
        users = self.get_all_users()
        users = [user for user in users if user.id != user_id]
        self._save_users([u.to_dict() for u in users])
    
    def _save_users(self, users_data):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=4, ensure_ascii=False) 