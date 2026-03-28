/**
 * API处理器 - 用于处理AI API调用，包含优雅的加载状态和错误处理
 */

class ApiHandler {
    constructor() {
        this.apiQueue = [];
        this.isProcessing = false;
        this.retryCount = 3;
        this.timeoutDuration = 120000; // 120秒超时
        this.injectStyles();
    }

    // 动态注入 CSS 文件
    injectStyles() {
        if (!document.getElementById('agentsart-loader-css')) {
            const style = document.createElement('style');
            style.id = 'agentsart-loader-css';
            style.innerHTML = `
                #neural-loading-overlay {
                    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    background: rgba(0, 5, 10, 0.9); backdrop-filter: blur(10px);
                    display: flex; flex-direction: column; justify-content: center; align-items: center;
                    z-index: 9999; color: #00e5ff; font-family: 'Segoe UI', monospace; text-transform: uppercase;
                }
                .loading-container { text-align: center; }
                .loading-text { font-size: 16px; letter-spacing: 4px; margin-bottom: 20px; animation: pulse 1.5s infinite alternate; text-shadow: 0 0 10px rgba(0, 229, 255, 0.5); }
                .loading-bar { width: 200px; height: 1px; background: rgba(0, 229, 255, 0.2); margin: 0 auto; overflow: hidden; position: relative; }
                .loading-bar .progress { position: absolute; width: 40%; height: 100%; background: #00e5ff; animation: scan 1.5s infinite ease-in-out; box-shadow: 0 0 10px #00e5ff; }
                #neural-loading-overlay.error-mode .loading-text { color: #ff5252; text-shadow: 0 0 10px rgba(255, 82, 82, 0.5); animation: none; }
                #neural-loading-overlay.error-mode .loading-bar .progress { background: #ff5252; animation: none; width: 100%; box-shadow: 0 0 10px #ff5252; }
                .error-indicator { color: #ff5252; margin-bottom: 20px; }
                .error-symbol { font-size: 48px; margin-bottom: 10px; text-shadow: 0 0 15px #ff5252; }
                .error-code { font-size: 12px; opacity: 0.7; margin-top: 10px; font-family: 'Courier New', monospace; }
                .auto-retry-info { font-size: 10px; color: #00e5ff; margin-top: 15px; letter-spacing: 2px; animation: pulse 1s infinite alternate; }
                .neural-network { margin-bottom: 30px; display: flex; justify-content: center; gap: 15px; }
                .neural-network .node { width: 8px; height: 8px; border-radius: 50%; background: #00e5ff; box-shadow: 0 0 10px #00e5ff; animation: blink 1s infinite alternate; }
                @keyframes pulse { 0% { opacity: 0.6; } 100% { opacity: 1; } }
                @keyframes blink { 0% { opacity: 0.2; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1.2); } }
                @keyframes scan { 0% { left: -40%; } 100% { left: 100%; } }
            `;
            document.head.appendChild(style);
        }
    }

    // 显示艺术化的加载状态
    showLoadingState(message = "CONNECTING TO NEURAL NETWORK...") {
        // 创建一个覆盖整个屏幕的加载层
        let loadingOverlay = document.getElementById('neural-loading-overlay');
        
        if (!loadingOverlay) {
            loadingOverlay = document.createElement('div');
            loadingOverlay.id = 'neural-loading-overlay';
            loadingOverlay.innerHTML = `
                <div class="loading-container">
                    <div class="neural-network">
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="node"></div>
                        <div class="connections"></div>
                    </div>
                    <div class="loading-text">${message}</div>
                    <div class="loading-bar">
                        <div class="progress"></div>
                    </div>
                    <div class="cyber-grid"></div>
                </div>
            `;
            
            document.body.appendChild(loadingOverlay);
        }
        
        // 更新文本
        loadingOverlay.querySelector('.loading-text').textContent = message;
        
        // 显示加载层
        loadingOverlay.style.display = 'flex';
        
        // 确保移除错误状态
        loadingOverlay.classList.remove('error-mode');
        const errorIndicator = loadingOverlay.querySelector('.error-indicator');
        if (errorIndicator) errorIndicator.remove();
    }

    // 隐藏加载状态
    hideLoadingState() {
        const loadingOverlay = document.getElementById('neural-loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
    
    // 清理错误状态，恢复到正常状态
    clearErrorState() {
        const loadingOverlay = document.getElementById('neural-loading-overlay');
        if (loadingOverlay) {
            // 移除错误指示器
            const errorIndicator = loadingOverlay.querySelector('.error-indicator');
            if (errorIndicator) {
                errorIndicator.remove();
            }
            loadingOverlay.classList.remove('error-mode');
        }
    }

    // 显示艺术化的错误状态（仅显示，不提供用户交互）
    showErrorState(errorMessage = "NEURAL NETWORK CONNECTION FAILED") {
        // 不隐藏加载状态，而是将其转换为错误状态
        this.showLoadingState(errorMessage);
        
        // 获取现有的加载层并修改其外观以表示错误状态
        let loadingOverlay = document.getElementById('neural-loading-overlay');
        if (loadingOverlay) {
            // 添加错误模式类
            loadingOverlay.classList.add('error-mode');
            
            // 修改样式以显示错误状态
            const container = loadingOverlay.querySelector('.loading-container');
            if (container) {
                // 添加错误指示器
                let errorIndicator = container.querySelector('.error-indicator');
                if (!errorIndicator) {
                    errorIndicator = document.createElement('div');
                    errorIndicator.className = 'error-indicator';
                    errorIndicator.innerHTML = `
                        <div class="error-symbol">✗</div>
                        <div class="error-message">${errorMessage}</div>
                        <div class="error-code">ERROR_CODE: 0x${Math.floor(Math.random()*16777215).toString(16).toUpperCase()}</div>
                        <div class="auto-retry-info">AUTO-RETRY IN PROGRESS...</div>
                    `;
                    container.insertBefore(errorIndicator, container.firstChild);
                } else {
                    errorIndicator.querySelector('.error-message').textContent = errorMessage;
                    errorIndicator.querySelector('.error-code').textContent = `ERROR_CODE: 0x${Math.floor(Math.random()*16777215).toString(16).toUpperCase()}`;
                }
                
                // 3秒后自动移除错误指示器，但保持加载状态以继续重试
                setTimeout(() => {
                    if (errorIndicator && errorIndicator.parentNode) {
                        errorIndicator.parentNode.removeChild(errorIndicator);
                        loadingOverlay.classList.remove('error-mode');
                    }
                }, 3000);
            }
        }
    }

    // API调用包装器
    async callApi(url, data, options = {}) {
        if (!options.silent) {
            this.showLoadingState(options.loadingMessage || "CONNECTING TO NEURAL NETWORK...");
        }
        
        try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), this.timeoutDuration);
            let response;
            try {
                response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    },
                    body: JSON.stringify(data),
                    signal: controller.signal
                });
            } finally {
                clearTimeout(timer);
            }
            
            if (!response.ok) {
                throw new Error(`API returned status: ${response.status}`);
            }
            
            // 检查返回类型
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                const text = await response.text();
                throw new Error(`Invalid response: ${text.substring(0, 100)}`);
            }
            
            const result = await response.json();
            if (!options.silent) {
                this.hideLoadingState();
            }
            return result;
            
        } catch (error) {
            console.error('API call failed:', error);
            
            // 根据错误类型显示不同的错误信息
            let errorMessage = "NEURAL NETWORK CONNECTION FAILED";
            if (error.name === 'AbortError' || error.message.includes('timeout')) {
                errorMessage = "NEURAL NETWORK TIMEOUT - SIGNAL LOST";
            } else if (error.message.includes('502')) {
                errorMessage = "NEURAL NETWORK GATEWAY ERROR";
            } else if (error.message.includes('429')) {
                errorMessage = "NEURAL NETWORK RATE LIMIT EXCEEDED";
            }
            
            if (!options.silent) {
                this.showErrorState(errorMessage);
            }
            throw error;
        }
    }
    
    // 带重试机制的API调用
    async callApiWithRetry(url, data, options = {}) {
        let lastError;
        
        for (let i = 0; i <= this.retryCount; i++) {
            try {
                if (i > 0) {
                    if (!options.silent) {
                        this.showLoadingState(`RETRYING CONNECTION (${i}/${this.retryCount})...`);
                    }
                    // 递增延迟重试
                    await new Promise(resolve => setTimeout(resolve, 2000 * i));
                }
                
                const result = await this.callApi(url, data, options);
                
                // 成功后清理错误状态
                this.clearErrorState();
                
                return result;
            } catch (error) {
                lastError = error;
                
                // 如果是最后一次重试，跳出循环
                if (i >= this.retryCount) {
                    break;
                }
            }
        }
        
        // 所有重试都失败了
        throw lastError;
    }
}

// 全局实例
window.AgentsArtApi = new ApiHandler();