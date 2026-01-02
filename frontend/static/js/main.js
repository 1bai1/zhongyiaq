/**
 * 系统全局JavaScript文件
 */

// 在文档加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 高亮当前活动的导航项
    highlightActiveNavItem();
    
    // 自动关闭警告提示
    setupAlertDismiss();
    
    // 设置下拉菜单
    setupDropdowns();
});

/**
 * 高亮当前页面对应的导航菜单项
 */
function highlightActiveNavItem() {
    // 获取当前页面的URL路径
    const currentPath = window.location.pathname;
    
    // 获取所有导航链接
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    // 遍历导航链接，检查是否匹配当前路径
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        
        // 如果链接的href属性不为空且当前路径以该href开头，则添加活动类
        if (href && currentPath.startsWith(href) && href !== '/') {
            link.classList.add('active');
            
            // 如果该链接在下拉菜单中，也高亮显示父级下拉菜单
            const dropdown = link.closest('.dropdown');
            if (dropdown) {
                const dropdownToggle = dropdown.querySelector('.dropdown-toggle');
                if (dropdownToggle) {
                    dropdownToggle.classList.add('active');
                }
            }
        }
    });
}

/**
 * 设置警告提示自动关闭
 */
function setupAlertDismiss() {
    // 获取所有警告提示
    const alerts = document.querySelectorAll('.alert');
    
    // 设置5秒后自动关闭
    alerts.forEach(alert => {
        // 创建Bootstrap的alert对象
        const bsAlert = new bootstrap.Alert(alert);
        
        // 5秒后关闭
        setTimeout(() => {
            bsAlert.close();
        }, 5000);
    });
}

/**
 * 设置Bootstrap下拉菜单
 */
function setupDropdowns() {
    // 获取所有下拉菜单触发器
    const dropdownToggleList = document.querySelectorAll('[data-bs-toggle="dropdown"]');
    
    // 为每个触发器创建下拉菜单实例
    const dropdownList = [...dropdownToggleList].map(dropdownToggleEl => {
        return new bootstrap.Dropdown(dropdownToggleEl);
    });
}

/**
 * 格式化日期时间
 * @param {string} dateTimeStr - 日期时间字符串
 * @returns {string} 格式化后的日期时间字符串
 */
function formatDateTime(dateTimeStr) {
    const date = new Date(dateTimeStr);
    
    // 格式化为 YYYY-MM-DD HH:MM:SS
    return `${date.getFullYear()}-${padZero(date.getMonth() + 1)}-${padZero(date.getDate())} ${padZero(date.getHours())}:${padZero(date.getMinutes())}:${padZero(date.getSeconds())}`;
}

/**
 * 数字补零
 * @param {number} num - 需要补零的数字
 * @returns {string} 补零后的字符串
 */
function padZero(num) {
    return num.toString().padStart(2, '0');
}

/**
 * 显示加载中状态
 * @param {HTMLElement} element - 显示加载中的元素
 * @param {string} message - 加载中显示的消息
 */
function showLoading(element, message = '加载中...') {
    element.innerHTML = `
        <div class="text-center my-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">加载中...</span>
            </div>
            <p class="mt-2">${message}</p>
        </div>
    `;
}

/**
 * 显示错误消息
 * @param {HTMLElement} element - 显示错误消息的元素
 * @param {string} message - 错误消息内容
 */
function showError(element, message) {
    element.innerHTML = `
        <div class="alert alert-danger" role="alert">
            <i class="bi bi-exclamation-triangle-fill me-2"></i>
            ${message}
        </div>
    `;
} 