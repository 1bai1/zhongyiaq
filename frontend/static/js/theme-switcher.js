/**
 * 主题切换器 - 中医药问答系统
 * 支持6种预设主题和自定义颜色
 */

// 可用主题列表
const themes = {
    'default': {
        name: '教育蓝',
        icon: '🔵',
        description: '默认主题，专业教育风格'
    },
    'tech-green': {
        name: '科技绿',
        icon: '🟢',
        description: '清新科技感，护眼绿色'
    },
    'vibrant-orange': {
        name: '活力橙',
        icon: '🟠',
        description: '充满活力，激发创造力'
    },
    'elegant-purple': {
        name: '优雅紫',
        icon: '🟣',
        description: '优雅高贵，艺术气息'
    },
    'midnight-blue': {
        name: '深夜蓝',
        icon: '🔷',
        description: '深邃稳重，专业范'
    },
    'rose-red': {
        name: '玫瑰红',
        icon: '🔴',
        description: '热情洋溢，充满朝气'
    },
    'dark': {
        name: '暗黑模式',
        icon: '🌙',
        description: '护眼暗黑，夜间使用'
    }
};

// 主题管理器
class ThemeManager {
    constructor() {
        this.currentTheme = this.getSavedTheme();
        this.init();
    }
    
    // 初始化
    init() {
        // 应用保存的主题
        this.applyTheme(this.currentTheme);
        
        // 添加主题切换器到页面
        this.addThemeSwitcher();
        
        // 监听主题变化
        this.setupEventListeners();
    }
    
    // 获取保存的主题
    getSavedTheme() {
        // 优先使用服务器保存的主题
        if (typeof current_user_theme !== 'undefined' && current_user_theme) {
            return current_user_theme;
        }
        // 否则使用本地存储
        return localStorage.getItem('user-theme') || 'default';
    }
    
    // 保存主题
    saveTheme(theme) {
        localStorage.setItem('user-theme', theme);
        
        // 如果用户已登录，同步到服务器
        if (typeof current_user_id !== 'undefined') {
            this.syncThemeToServer(theme);
        }
    }
    
    // 同步主题到服务器
    syncThemeToServer(theme) {
        fetch('/api/user/theme', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ theme: theme })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('主题已同步到服务器');
            }
        })
        .catch(error => {
            console.error('同步主题失败:', error);
        });
    }
    
    // 应用主题
    applyTheme(theme) {
        // 移除旧主题
        document.documentElement.removeAttribute('data-theme');
        
        // 应用新主题
        if (theme !== 'default') {
            document.documentElement.setAttribute('data-theme', theme);
        }
        
        this.currentTheme = theme;
        this.saveTheme(theme);
        
        // 触发主题变化事件
        document.dispatchEvent(new CustomEvent('theme-changed', { 
            detail: { theme: theme } 
        }));
    }
    
    // 添加主题切换器到页面
    addThemeSwitcher() {
        // 如果已存在，不重复添加
        if (document.getElementById('theme-switcher-container')) {
            return;
        }
        
        const navbar = document.querySelector('.navbar-right');
        if (!navbar) return;
        
        const switcherHTML = `
            <div id="theme-switcher-container" class="dropdown me-3">
                <button class="btn btn-outline-secondary btn-sm dropdown-toggle" 
                        type="button" id="themeSwitcher" 
                        data-bs-toggle="dropdown" 
                        aria-expanded="false"
                        title="切换主题">
                    <i class="bi bi-palette"></i>
                    <span class="d-none d-md-inline ms-1">主题</span>
                </button>
                <ul class="dropdown-menu dropdown-menu-end theme-menu" aria-labelledby="themeSwitcher">
                    <li class="dropdown-header">选择颜色主题</li>
                    <li><hr class="dropdown-divider"></li>
                    ${this.generateThemeMenuItems()}
                </ul>
            </div>
        `;
        
        navbar.insertAdjacentHTML('afterbegin', switcherHTML);
    }
    
    // 生成主题菜单项
    generateThemeMenuItems() {
        let html = '';
        for (const [key, value] of Object.entries(themes)) {
            const isActive = this.currentTheme === key ? 'active' : '';
            html += `
                <li>
                    <a class="dropdown-item theme-option ${isActive}" 
                       href="#" 
                       data-theme="${key}">
                        <span class="theme-icon">${value.icon}</span>
                        <span class="theme-name">${value.name}</span>
                        ${isActive ? '<i class="bi bi-check2 float-end"></i>' : ''}
                    </a>
                </li>
            `;
        }
        return html;
    }
    
    // 设置事件监听
    setupEventListeners() {
        // 监听主题选项点击
        document.addEventListener('click', (e) => {
            if (e.target.closest('.theme-option')) {
                e.preventDefault();
                const theme = e.target.closest('.theme-option').dataset.theme;
                this.applyTheme(theme);
                
                // 更新菜单选中状态
                this.updateMenuSelection();
                
                // 显示提示
                this.showThemeChangeToast(themes[theme].name);
            }
        });
    }
    
    // 更新菜单选中状态
    updateMenuSelection() {
        document.querySelectorAll('.theme-option').forEach(item => {
            const theme = item.dataset.theme;
            if (theme === this.currentTheme) {
                item.classList.add('active');
                if (!item.querySelector('.bi-check2')) {
                    item.innerHTML += '<i class="bi bi-check2 float-end"></i>';
                }
            } else {
                item.classList.remove('active');
                const check = item.querySelector('.bi-check2');
                if (check) check.remove();
            }
        });
    }
    
    // 显示主题切换提示
    showThemeChangeToast(themeName) {
        // 创建简单的toast提示
        const toast = document.createElement('div');
        toast.className = 'theme-change-toast';
        toast.innerHTML = `
            <i class="bi bi-palette me-2"></i>
            已切换到 <strong>${themeName}</strong>
        `;
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--primary-color);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: var(--shadow-lg);
            z-index: 9999;
            animation: slideInUp 0.3s ease;
        `;
        
        document.body.appendChild(toast);
        
        // 3秒后移除
        setTimeout(() => {
            toast.style.animation = 'slideOutDown 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
}

// 添加动画CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInUp {
        from {
            transform: translateY(100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutDown {
        from {
            transform: translateY(0);
            opacity: 1;
        }
        to {
            transform: translateY(100%);
            opacity: 0;
        }
    }
    
    .theme-menu {
        min-width: 220px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .theme-option {
        display: flex;
        align-items: center;
        padding: 8px 16px;
        cursor: pointer;
    }
    
    .theme-option:hover {
        background-color: var(--light-color);
    }
    
    .theme-option.active {
        background-color: var(--primary-color);
        color: white !important;
    }
    
    .theme-icon {
        font-size: 1.2em;
        margin-right: 10px;
    }
    
    .theme-name {
        flex: 1;
    }
`;
document.head.appendChild(style);

// 页面加载完成后初始化主题管理器
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.themeManager = new ThemeManager();
    });
} else {
    window.themeManager = new ThemeManager();
}

