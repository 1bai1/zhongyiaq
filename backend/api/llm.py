from flask import Blueprint, render_template, request, jsonify, current_app, session, Response
from flask_login import login_required, current_user
from backend.utils.log_manager import LogManager
from backend.utils.helpers import get_client_ip
import json
import os
import time
import requests
import sys
from pathlib import Path

# 添加RAG-proj模块到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_proj_dir = os.path.join(os.path.dirname(current_dir), 'RAG-proj')
sys.path.append(rag_proj_dir)

# 导入RAG系统
try:
    from rag_backend import RAGSystem
    RAG_AVAILABLE = True
    print("RAG系统导入成功")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"RAG系统导入失败: {e}")
    RAGSystem = None

llm_bp = Blueprint('llm', __name__)

# 保存对话历史的目录
CONVERSATIONS_DIR = 'data/conversations'
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

# API配置
API_URL = "YOUR_API_URL"
API_KEY = "YOUR_API_KEY"
API_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

# 全局RAG系统实例
rag_system = None

def initialize_rag_system():
    """初始化RAG系统"""
    global rag_system
    if not RAG_AVAILABLE:
        print("RAG系统不可用，跳过初始化")
        return False
    
    try:
        print("正在初始化RAG系统...")
        rag_system = RAGSystem(verbose=True)
        rag_system.load_model()
        rag_system.load_documents()
        rag_system.build_index()
        print("RAG系统初始化完成")
        return True
    except Exception as e:
        print(f"RAG系统初始化失败: {e}")
        rag_system = None
        return False

def get_rag_context(query, top_k=3):
    """获取RAG检索上下文"""
    if not rag_system:
        print(f"[RAG] RAG系统未初始化，跳过检索")
        return ""
    
    try:
        print(f"[RAG] 正在检索查询: {query}")
        results = rag_system.search(query, top_k=top_k)
        if not results:
            print(f"[RAG] 未找到相关结果")
            return ""
        
        print(f"[RAG] 找到 {len(results)} 个相关结果:")
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            print(f"[RAG] 结果 {i} (相似度: {score:.3f}):")
            print(f"[RAG] {doc[:200]}{'...' if len(doc) > 200 else ''}")
            print(f"[RAG] " + "="*50)
            context_parts.append(f"参考资料{i}（相似度: {score:.3f}）:\n{doc}")
        
        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"[RAG] RAG检索出错: {e}")
        return ""

@llm_bp.route('/chat')
@login_required
def chat_page():
    """显示聊天页面"""
    return render_template('chat.html')

@llm_bp.route('/api/chat/send', methods=['POST'])
@login_required
def send_message():
    """发送消息到LLM服务"""
    data = request.json
    message = data.get('message', '')
    conversation_id = data.get('conversation_id', f"conv_{int(time.time())}")
    
    if not message:
        return jsonify({'error': '消息不能为空'}), 400
    
    # 获取对话历史
    conversation = load_conversation(conversation_id, current_user.id)
    
    # 添加用户消息
    conversation['messages'].append({
        'role': 'user',
        'content': message,
        'timestamp': time.time()
    })
    
    # 保存对话
    save_conversation(conversation_id, current_user.id, conversation)
    
    # 记录聊天日志
    log_manager = LogManager()
    log_manager.create_log(
        user_id=current_user.id,
        username=current_user.username,
        ip=get_client_ip(),
        action='chat_message',
        details={
            'conversation_id': conversation_id,
            'message_type': 'user'
        }
    )
    
    # 调用LLM API
    api_messages = []
    for msg in conversation['messages'][-10:]:  # 最多发送最近10条消息作为上下文
        if msg['role'] in ['user', 'assistant']:
            api_messages.append({
                'role': msg['role'],
                'content': msg['content']
            })
    
    # 使用流式响应收集完整回答
    full_response = ""
    for chunk in generate_llm_stream_response(api_messages, user_query=message):
        if chunk:
            full_response += chunk
    
    # 添加LLM响应
    conversation['messages'].append({
        'role': 'assistant',
        'content': full_response,
        'timestamp': time.time()
    })
    
    # 保存更新后的对话
    save_conversation(conversation_id, current_user.id, conversation)
    
    return jsonify({
        'success': True,
        'response': full_response,
        'conversation_id': conversation_id
    })

@llm_bp.route('/api/chat/stream', methods=['POST'])
@login_required
def stream_message():
    """流式发送消息到LLM服务"""
    data = request.json
    message = data.get('message', '')
    conversation_id = data.get('conversation_id', f"conv_{int(time.time())}")
    
    if not message:
        return jsonify({'error': '消息不能为空'}), 400
    
    # 获取对话历史
    conversation = load_conversation(conversation_id, current_user.id)
    
    # 添加用户消息
    conversation['messages'].append({
        'role': 'user',
        'content': message,
        'timestamp': time.time()
    })
    
    # 保存对话
    save_conversation(conversation_id, current_user.id, conversation)
    
    # 记录聊天日志
    log_manager = LogManager()
    log_manager.create_log(
        user_id=current_user.id,
        username=current_user.username,
        ip=get_client_ip(),
        action='chat_message_stream',
        details={
            'conversation_id': conversation_id,
            'message_type': 'user'
        }
    )
    
    def generate():
        try:
            # 准备API消息
            api_messages = []
            for msg in conversation['messages'][-10:]:
                if msg['role'] in ['user', 'assistant']:
                    api_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            # 调用流式LLM API
            full_response = ""
            for chunk in generate_llm_stream_response(api_messages, user_query=message):
                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk, 'conversation_id': conversation_id})}\n\n"
            
            # 发送完成信号
            yield f"data: {json.dumps({'done': True, 'conversation_id': conversation_id})}\n\n"
            
            # 保存完整响应到对话历史
            conversation['messages'].append({
                'role': 'assistant',
                'content': full_response,
                'timestamp': time.time()
            })
            save_conversation(conversation_id, current_user.id, conversation)
            
        except Exception as e:
            error_msg = f"流式响应出错: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
    })

@llm_bp.route('/api/chat/history/<conversation_id>', methods=['GET'])
@login_required
def get_chat_history(conversation_id):
    """获取特定对话的历史记录"""
    conversation = load_conversation(conversation_id, current_user.id)
    return jsonify(conversation)

@llm_bp.route('/api/chat/conversations', methods=['GET'])
@login_required
def get_conversations():
    """获取用户的所有对话列表"""
    user_conversations_dir = os.path.join(CONVERSATIONS_DIR, current_user.id)
    
    if not os.path.exists(user_conversations_dir):
        return jsonify([])
    
    conversations = []
    for filename in os.listdir(user_conversations_dir):
        if filename.endswith('.json'):
            conversation_id = filename[:-5]  # 移除.json扩展名
            conversation = load_conversation(conversation_id, current_user.id)
            
            # 提取对话摘要信息
            summary = {
                'id': conversation_id,
                'title': conversation.get('title', f"对话 {conversation_id}"),
                'created_at': conversation.get('created_at', 0),
                'updated_at': conversation.get('updated_at', 0),
                'message_count': len(conversation.get('messages', []))
            }
            
            conversations.append(summary)
    
    # 按更新时间排序
    conversations.sort(key=lambda x: x['updated_at'], reverse=True)
    return jsonify(conversations)

@llm_bp.route('/api/chat/conversation/<conversation_id>', methods=['DELETE'])
@login_required
def delete_conversation(conversation_id):
    """删除特定对话"""
    conversation_file = os.path.join(CONVERSATIONS_DIR, current_user.id, f"{conversation_id}.json")
    
    if not os.path.exists(conversation_file):
        return jsonify({'error': '对话不存在'}), 404
    
    try:
        os.remove(conversation_file)
        
        # 记录删除日志
        log_manager = LogManager()
        log_manager.create_log(
            user_id=current_user.id,
            username=current_user.username,
            ip=get_client_ip(),
            action='delete_conversation',
            details={'conversation_id': conversation_id}
        )
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'删除失败: {str(e)}'}), 500

def load_conversation(conversation_id, user_id):
    """加载对话历史"""
    user_conversations_dir = os.path.join(CONVERSATIONS_DIR, user_id)
    os.makedirs(user_conversations_dir, exist_ok=True)
    
    conversation_file = os.path.join(user_conversations_dir, f"{conversation_id}.json")
    
    if os.path.exists(conversation_file):
        try:
            with open(conversation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    
    # 如果文件不存在或读取失败，创建新对话
    return {
        'id': conversation_id,
        'title': f"新对话 {time.strftime('%Y-%m-%d %H:%M')}",
        'created_at': time.time(),
        'updated_at': time.time(),
        'messages': []
    }

def save_conversation(conversation_id, user_id, conversation):
    """保存对话历史"""
    user_conversations_dir = os.path.join(CONVERSATIONS_DIR, user_id)
    os.makedirs(user_conversations_dir, exist_ok=True)
    
    conversation_file = os.path.join(user_conversations_dir, f"{conversation_id}.json")
    
    # 更新时间戳
    conversation['updated_at'] = time.time()
    
    with open(conversation_file, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

def generate_llm_stream_response(messages, user_query=None):
    """调用LLM API生成流式响应"""
    try:
        # 获取RAG检索上下文
        rag_context = ""
        if user_query and rag_system:
            rag_context = get_rag_context(user_query, top_k=3)
        
        # 构建系统提示词，包含RAG上下文
        system_content = """你是一位资深的中医药专家和智能助手，专门为用户提供中草药识别和中医药知识咨询服务。

## 🌿 专业背景
- **深厚学识**：精通中医药理论，熟悉《本草纲目》、《神农本草经》等经典著作
- **实践经验**：了解中草药的形态特征、生长环境、采收加工和质量鉴别
- **现代融合**：结合传统中医理论与现代药理研究成果

## 🎯 核心服务
### 中草药识别支持
- 协助分析中草药的外观特征和识别要点
- 解释药材的真伪鉴别方法
- 介绍药材的产地、采收时间等相关信息

### 中医药知识普及
- 详细介绍中药材的性味归经、功效主治
- 解释中医基础理论（如四气五味、升降浮沉等）
- 分享中药配伍原理和经典方剂知识

### 系统使用指导
- 说明中草药识别系统的功能和使用方法
- 解答技术操作相关问题
- 提供学习建议和资源推荐

## 💬 交流风格
- **专业严谨**：确保所有信息准确可靠，有据可查
- **通俗易懂**：用生动的比喻和简单的语言解释复杂概念
- **耐心细致**：详细回答每个问题，不厌其烦地解释疑惑
- **温和友善**：保持亲切的语调，让用户感到舒适和信任

## ⚠️ 重要声明
- **学习参考**：所提供信息仅供学习和参考，不构成医疗建议
- **专业就医**：任何疾病诊断和治疗请咨询专业中医师
- **安全第一**：强调中药使用需在专业指导下进行
- **识别辅助**：系统识别结果仅供参考，最终确认需专业人士

## 🚫 服务边界
- 不进行疾病诊断或开具处方
- 不推荐具体的治疗方案
- 不替代专业医疗咨询
- 不保证识别结果的绝对准确性

请放心向我咨询任何中医药相关问题，我将竭诚为您提供专业、可靠的知识服务！"""

        # 如果有RAG检索结果，添加到系统提示词中
        if rag_context:
            system_content += f"""

以下是从中医药知识库中检索到的相关信息，请结合这些信息来回答用户的问题：

{rag_context}

请基于以上检索到的专业资料，结合你的中医药知识，为用户提供准确、专业的回答。如果检索结果与问题相关，请优先使用检索到的信息。"""

        system_prompt = {
            "role": "system",
            "content": system_content
        }
        
        # 将系统提示词添加到消息列表开头
        api_messages = [system_prompt] + messages
        
        payload = {
            "model": API_MODEL,
            "stream": True,  # 启用流式响应
            "max_tokens": 51200,
            "min_p": 0.05,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "stop": [],
            "messages": api_messages
        }
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120, stream=True)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"流式API调用失败，状态码: {response.status_code}, 内容: {response.text}")
            yield f"API调用失败，状态码: {response.status_code}"
    
    except Exception as e:
        print(f"调用流式LLM API时出错: {str(e)}")
        yield f"抱歉，系统出现错误: {str(e)}"