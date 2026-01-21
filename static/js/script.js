// SENTINEL - Silence Index Investigation System
// Version 2.0 - Forensic AI Chat

const API_BASE = 'http://localhost:5000/api';

// State
let currentSession = null;
let dynamicChart = null;
let overviewChart = null;
let genderChart = null;
let casteChart = null;
let incomeChart = null;
let categoryChart = null;
let temporalChart = null;

// Chart.js defaults for dark theme
Chart.defaults.color = '#a0a0b0';
Chart.defaults.borderColor = '#2a2a3a';

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('[SENTINEL] Silence Index Investigation System');
    
    // Update timestamp
    updateTimestamp();
    setInterval(updateTimestamp, 1000);
    
    // Navigation
    initNavigation();
    
    // Load initial data
    loadDashboard();
    loadSessions();
    
    // Chat functionality
    initChat();
    
    // Search functionality
    initSearch();
    
    // Report functionality
    initReport();
    
    // Mobile menu
    initMobileMenu();
});

function updateTimestamp() {
    const now = new Date();
    const timestamp = now.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
    document.getElementById('timestamp').textContent = timestamp;
}

// ============================================================
// NAVIGATION
// ============================================================

function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const screens = document.querySelectorAll('.screen');
    
    const pageNames = {
        'dashboard': 'Command Center',
        'chat': 'AI Analysis',
        'search': 'Case Search',
        'demographics': 'Demographics',
        'geography': 'Geographic Analysis',
        'categories': 'Category Analysis',
        'temporal': 'Temporal Analysis',
        'report': 'Full Report'
    };
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const screenId = item.dataset.screen;
            
            // Update nav
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // Update screen
            screens.forEach(s => s.classList.remove('active'));
            document.getElementById(screenId)?.classList.add('active');
            
            // Update header title
            document.getElementById('pageTitle').textContent = pageNames[screenId] || screenId;
            
            // Load screen data
            loadScreenData(screenId);
            
            // Close mobile menu
            document.querySelector('.sidebar')?.classList.remove('open');
        });
    });
}

function loadScreenData(screenId) {
    switch(screenId) {
        case 'dashboard':
            loadDashboard();
            break;
        case 'demographics':
            loadDemographics();
            break;
        case 'geography':
            loadGeography();
            break;
        case 'categories':
            loadCategories();
            break;
        case 'temporal':
            loadTemporal();
            break;
    }
}

function initMobileMenu() {
    const menuBtn = document.getElementById('menuBtn');
    const sidebar = document.querySelector('.sidebar');
    
    menuBtn?.addEventListener('click', () => {
        sidebar?.classList.toggle('open');
    });
}

// ============================================================
// DASHBOARD
// ============================================================

async function loadDashboard() {
    try {
        const response = await fetch(`${API_BASE}/stats`);
        const result = await response.json();
        
        if (result.success) {
            const data = result.data;
            
            // Update stat cards
            document.getElementById('totalCases').textContent = data.total_complaints.toLocaleString();
            document.getElementById('headerCases').textContent = data.total_complaints.toLocaleString();
            
            const silenced = Math.round(data.total_complaints * data.silence_rate / 100);
            document.getElementById('silencedCount').textContent = silenced.toLocaleString();
            document.getElementById('silenceRate').textContent = data.silence_rate + '%';
            document.getElementById('headerSilenced').textContent = silenced.toLocaleString();
            
            document.getElementById('avgScore').textContent = data.avg_silence_score;
            document.getElementById('avgDays').textContent = Math.round(data.avg_days_in_system);
            
            // Update alert
            document.getElementById('alertText').textContent = 
                `${data.silence_rate}% of complaints (${silenced.toLocaleString()} cases) are being systematically ignored`;
            
            // Load overview chart
            loadOverviewChart();
        }
    } catch (error) {
        console.error('Dashboard error:', error);
    }
}

async function loadOverviewChart() {
    try {
        const response = await fetch(`${API_BASE}/demographic-silence`);
        const result = await response.json();
        
        if (result.success) {
            const ctx = document.getElementById('overviewChart');
            if (!ctx) return;
            
            if (overviewChart) overviewChart.destroy();
            
            const incomeData = result.data.by_income;
            const labels = ['0-3L', '3-6L', '6-10L', '10L+'];
            const scores = labels.map(l => incomeData[l]?.avg_silence || 0);
            
            overviewChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Avg Silence Score',
                        data: scores,
                        backgroundColor: ['#ff6b6b', '#ffb700', '#00d4ff', '#00ff88'],
                        borderWidth: 0,
                        borderRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        title: {
                            display: true,
                            text: 'Silence Score by Income Level',
                            color: '#a0a0b0',
                            font: { size: 12, weight: 500 }
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            grid: { color: '#2a2a3a' }
                        },
                        x: {
                            grid: { display: false }
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Overview chart error:', error);
    }
}

// Quick analysis from dashboard
function quickAnalysis(query) {
    // Navigate to chat
    document.querySelector('[data-screen="chat"]')?.click();
    
    // Send the query
    setTimeout(() => {
        sendChatMessage(query);
    }, 300);
}

// ============================================================
// CHAT FUNCTIONALITY
// ============================================================

function initChat() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    const newSessionBtn = document.getElementById('newSessionBtn');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const sessionSelect = document.getElementById('sessionSelect');
    
    // Send message
    sendBtn?.addEventListener('click', () => {
        const msg = chatInput?.value.trim();
        if (msg) sendChatMessage(msg);
    });
    
    chatInput?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const msg = chatInput.value.trim();
            if (msg) sendChatMessage(msg);
        }
    });
    
    // New session
    newSessionBtn?.addEventListener('click', createNewSession);
    
    // Clear chat display
    clearChatBtn?.addEventListener('click', () => {
        const messages = document.getElementById('chatMessages');
        messages.innerHTML = `
            <div class="message system">
                <div class="message-content">
                    <p><strong>SENTINEL INITIALIZED</strong></p>
                    <p>Chat display cleared. Session history preserved.</p>
                </div>
            </div>
        `;
    });
    
    // Session select
    sessionSelect?.addEventListener('change', (e) => {
        const sessionId = e.target.value;
        if (sessionId) {
            loadSessionHistory(sessionId);
        } else {
            currentSession = null;
        }
    });
}

async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE}/chat/sessions`);
        const result = await response.json();
        
        if (result.success) {
            const select = document.getElementById('sessionSelect');
            if (!select) return;
            
            select.innerHTML = '<option value="">New Investigation</option>';
            
            result.data.sessions.forEach(session => {
                const option = document.createElement('option');
                option.value = session.session_id;
                option.textContent = session.name || `Session ${session.session_id.slice(0, 8)}`;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Load sessions error:', error);
    }
}

async function createNewSession() {
    const name = prompt('Enter session name:', `Investigation ${new Date().toLocaleDateString()}`);
    if (!name) return;
    
    try {
        const response = await fetch(`${API_BASE}/chat/session/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentSession = result.data.session_id;
            await loadSessions();
            
            const select = document.getElementById('sessionSelect');
            if (select) select.value = currentSession;
            
            // Clear chat display
            const messages = document.getElementById('chatMessages');
            messages.innerHTML = `
                <div class="message system">
                    <div class="message-content">
                        <p><strong>NEW SESSION: ${name}</strong></p>
                        <p>Session ID: ${currentSession.slice(0, 8)}...</p>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        console.error('Create session error:', error);
    }
}

async function loadSessionHistory(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/chat/history/${sessionId}`);
        const result = await response.json();
        
        if (result.success) {
            currentSession = sessionId;
            
            const messages = document.getElementById('chatMessages');
            messages.innerHTML = '';
            
            result.data.messages.forEach(msg => {
                addMessageToChat(msg.role, msg.content);
            });
            
            messages.scrollTop = messages.scrollHeight;
        }
    } catch (error) {
        console.error('Load history error:', error);
    }
}

async function sendChatMessage(message) {
    const chatInput = document.getElementById('chatInput');
    const messagesDiv = document.getElementById('chatMessages');
    
    // Clear input
    if (chatInput) chatInput.value = '';
    
    // Add user message
    addMessageToChat('user', message);
    
    // Add typing indicator
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                session_id: currentSession
            })
        });
        
        const result = await response.json();
        
        // Remove typing indicator
        typingDiv.remove();
        
        if (result.success) {
            // Update session if new
            if (result.data.session_id && !currentSession) {
                currentSession = result.data.session_id;
                await loadSessions();
            }
            
            // Add assistant message
            addMessageToChat('assistant', result.data.response);
            
            // Handle chart if present
            if (result.data.chart_data) {
                renderDynamicChart(result.data.chart_data);
            }
        } else {
            addMessageToChat('assistant', `Error: ${result.error}`);
        }
    } catch (error) {
        typingDiv.remove();
        addMessageToChat('assistant', `Connection error: ${error.message}`);
    }
}

function addMessageToChat(role, content) {
    const messagesDiv = document.getElementById('chatMessages');
    
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    // Parse markdown if available
    let html = content;
    if (typeof marked !== 'undefined') {
        try {
            html = marked.parse(content);
        } catch (e) {
            html = content;
        }
    }
    
    msgDiv.innerHTML = `<div class="message-content">${html}</div>`;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function sendSuggestion(query) {
    sendChatMessage(query);
}

// ============================================================
// DYNAMIC CHART
// ============================================================

function renderDynamicChart(chartData) {
    const panel = document.getElementById('chartPanel');
    const titleEl = document.getElementById('chartTitle');
    const ctx = document.getElementById('dynamicChart');
    
    if (!panel || !ctx) return;
    
    panel.style.display = 'flex';
    titleEl.textContent = chartData.title || 'Analysis Chart';
    
    if (dynamicChart) dynamicChart.destroy();
    
    const colors = ['#00d4ff', '#ffb700', '#ff6b6b', '#00ff88', '#a855f7', '#ec4899'];
    
    dynamicChart = new Chart(ctx, {
        type: chartData.type || 'bar',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: chartData.label || 'Value',
                data: chartData.values,
                backgroundColor: colors.slice(0, chartData.labels.length),
                borderWidth: 0,
                borderRadius: chartData.type === 'bar' ? 2 : 0,
                tension: chartData.type === 'line' ? 0.4 : 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: chartData.type !== 'pie' && chartData.type !== 'doughnut' ? {
                y: {
                    beginAtZero: true,
                    grid: { color: '#2a2a3a' }
                },
                x: {
                    grid: { display: false }
                }
            } : undefined
        }
    });
}

function closeChart() {
    const panel = document.getElementById('chartPanel');
    if (panel) panel.style.display = 'none';
    if (dynamicChart) {
        dynamicChart.destroy();
        dynamicChart = null;
    }
}

// ============================================================
// SEARCH
// ============================================================

function initSearch() {
    const searchBtn = document.getElementById('searchBtn');
    const searchInput = document.getElementById('searchInput');
    
    searchBtn?.addEventListener('click', performSearch);
    searchInput?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });
}

async function performSearch() {
    const query = document.getElementById('searchInput')?.value.trim();
    const silencedOnly = document.getElementById('silencedOnly')?.checked;
    const resultsPanel = document.getElementById('searchResults');
    
    if (!query) {
        alert('Please enter a search query');
        return;
    }
    
    resultsPanel.querySelector('.panel-body').innerHTML = `
        <div class="empty-state">
            <div class="spinner"></div>
            <p>Searching...</p>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: 20,
                silence_threshold: silencedOnly ? 70 : null
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            const data = result.data;
            
            if (data.results.length === 0) {
                resultsPanel.querySelector('.panel-body').innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-search"></i>
                        <p>No results found</p>
                    </div>
                `;
                return;
            }
            
            let html = '';
            data.results.forEach(r => {
                const scoreClass = r.silence_score >= 70 ? 'high' : (r.silence_score >= 50 ? 'medium' : 'low');
                
                html += `
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-id">#${r.id || 'N/A'}</span>
                            <span class="result-score ${scoreClass}">${r.silence_score}</span>
                        </div>
                        <div class="result-text">${r.text}</div>
                        <div class="result-meta">
                            <span><i class="fas fa-tag"></i> ${r.category}</span>
                            <span><i class="fas fa-user"></i> ${r.gender}</span>
                            <span><i class="fas fa-wallet"></i> ${r.income}</span>
                            <span><i class="fas fa-map-marker-alt"></i> ${r.ward}</span>
                        </div>
                    </div>
                `;
            });
            
            resultsPanel.querySelector('.panel-body').innerHTML = html;
        }
    } catch (error) {
        resultsPanel.querySelector('.panel-body').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error: ${error.message}</p>
            </div>
        `;
    }
}

// ============================================================
// DEMOGRAPHICS
// ============================================================

async function loadDemographics() {
    try {
        const response = await fetch(`${API_BASE}/demographic-silence`);
        const result = await response.json();
        
        if (result.success) {
            const data = result.data;
            
            // Gender chart
            createDemoChart('genderChart', data.by_gender, 'gender', genderChart, (c) => genderChart = c);
            
            // Caste chart
            createDemoChart('casteChart', data.by_caste, 'caste', casteChart, (c) => casteChart = c);
            
            // Income chart
            const incomeOrder = ['0-3L', '3-6L', '6-10L', '10L+'];
            const incomeData = {};
            incomeOrder.forEach(k => {
                if (data.by_income[k]) incomeData[k] = data.by_income[k];
            });
            createDemoChart('incomeChart', incomeData, 'income', incomeChart, (c) => incomeChart = c);
            
            // Findings
            const findings = document.getElementById('demoFindings');
            if (findings) {
                findings.innerHTML = '';
                
                // Gender disparity
                if (data.by_gender.F && data.by_gender.M) {
                    const ratio = (data.by_gender.F.avg_silence / data.by_gender.M.avg_silence).toFixed(2);
                    findings.innerHTML += `<li>Women face <strong>${ratio}x</strong> higher silence scores than men</li>`;
                }
                
                // Income disparity
                if (data.by_income['0-3L'] && data.by_income['10L+']) {
                    const ratio = (data.by_income['0-3L'].avg_silence / data.by_income['10L+'].avg_silence).toFixed(2);
                    findings.innerHTML += `<li>Lowest income bracket silenced <strong>${ratio}x</strong> more than highest</li>`;
                }
                
                // Caste disparity
                if (data.by_caste.SC && data.by_caste.General) {
                    const ratio = (data.by_caste.SC.avg_silence / data.by_caste.General.avg_silence).toFixed(2);
                    findings.innerHTML += `<li>SC caste faces <strong>${ratio}x</strong> higher silence rates than General</li>`;
                }
            }
        }
    } catch (error) {
        console.error('Demographics error:', error);
    }
}

function createDemoChart(canvasId, data, type, chartRef, setChart) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    if (chartRef) chartRef.destroy();
    
    const labels = Object.keys(data);
    const values = labels.map(l => data[l].avg_silence);
    const colors = {
        'gender': ['#ff6b6b', '#00d4ff', '#ffb700'],
        'caste': ['#00ff88', '#ffb700', '#ff6b6b', '#a855f7'],
        'income': ['#ff6b6b', '#ffb700', '#00d4ff', '#00ff88']
    };
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors[type] || ['#00d4ff'],
                borderWidth: 0,
                borderRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, max: 100, grid: { color: '#2a2a3a' } },
                x: { grid: { display: false } }
            }
        }
    });
    
    setChart(chart);
}

// ============================================================
// GEOGRAPHY
// ============================================================

async function loadGeography() {
    try {
        const response = await fetch(`${API_BASE}/geographic-silence?top_n=15`);
        const result = await response.json();
        
        if (result.success) {
            const wards = result.data.top_silenced;
            const tbody = document.getElementById('geoTableBody');
            
            if (tbody) {
                tbody.innerHTML = '';
                
                wards.forEach((ward, i) => {
                    const statusClass = ward.avg_silence >= 70 ? 'critical' : (ward.avg_silence >= 50 ? 'warning' : 'ok');
                    const statusText = ward.avg_silence >= 70 ? 'CRITICAL' : (ward.avg_silence >= 50 ? 'WARNING' : 'OK');
                    
                    tbody.innerHTML += `
                        <tr>
                            <td>${i + 1}</td>
                            <td>${ward.ward}</td>
                            <td>${ward.avg_silence}</td>
                            <td>${ward.count}</td>
                            <td>${ward.silenced_pct}%</td>
                            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
                        </tr>
                    `;
                });
            }
        }
    } catch (error) {
        console.error('Geography error:', error);
    }
}

// ============================================================
// CATEGORIES
// ============================================================

async function loadCategories() {
    try {
        const response = await fetch(`${API_BASE}/complaint-types`);
        const result = await response.json();
        
        if (result.success) {
            const categories = result.data;
            const ctx = document.getElementById('categoryChart');
            
            if (ctx) {
                if (categoryChart) categoryChart.destroy();
                
                categoryChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: categories.map(c => c.category),
                        datasets: [{
                            label: 'Silenced %',
                            data: categories.map(c => c.silenced_pct),
                            backgroundColor: '#ff6b6b',
                            borderWidth: 0,
                            borderRadius: 2
                        }]
                    },
                    options: {
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { beginAtZero: true, max: 100, grid: { color: '#2a2a3a' } },
                            y: { grid: { display: false } }
                        }
                    }
                });
            }
            
            // Findings
            const findings = document.getElementById('catFindings');
            if (findings) {
                const most = categories[0];
                const least = categories[categories.length - 1];
                
                findings.innerHTML = `
                    <li><strong>${most.category}</strong> complaints are most ignored (${most.silenced_pct}% silenced)</li>
                    <li><strong>${least.category}</strong> complaints are least ignored (${least.silenced_pct}% silenced)</li>
                    <li>Critical infrastructure issues face systematic deprioritization</li>
                `;
            }
        }
    } catch (error) {
        console.error('Categories error:', error);
    }
}

// ============================================================
// TEMPORAL
// ============================================================

async function loadTemporal() {
    try {
        const response = await fetch(`${API_BASE}/temporal-decay`);
        const result = await response.json();
        
        if (result.success) {
            const buckets = result.data;
            const ctx = document.getElementById('temporalChart');
            
            if (ctx) {
                if (temporalChart) temporalChart.destroy();
                
                temporalChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: buckets.map(b => b.time_bucket),
                        datasets: [{
                            label: 'Silenced %',
                            data: buckets.map(b => b.silenced_pct),
                            borderColor: '#ff6b6b',
                            backgroundColor: 'rgba(255, 107, 107, 0.1)',
                            fill: true,
                            tension: 0.4,
                            borderWidth: 2
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: { beginAtZero: true, max: 100, grid: { color: '#2a2a3a' } },
                            x: { grid: { display: false } }
                        }
                    }
                });
            }
            
            // Insight
            const insight = document.getElementById('temporalInsight');
            if (insight && buckets.length > 0) {
                const first = buckets[0];
                const last = buckets[buckets.length - 1];
                
                insight.textContent = `Complaints in the ${first.time_bucket} range have ${first.silenced_pct}% silence rate. Those in ${last.time_bucket} range reach ${last.silenced_pct}%. This exponential decay demonstrates institutional neglect - the longer a complaint waits, the more likely it's forgotten.`;
            }
        }
    } catch (error) {
        console.error('Temporal error:', error);
    }
}

// ============================================================
// REPORT
// ============================================================

function initReport() {
    const runBtn = document.getElementById('runReportBtn');
    runBtn?.addEventListener('click', runFullReport);
}

async function runFullReport() {
    const runBtn = document.getElementById('runReportBtn');
    const statusPanel = document.getElementById('reportStatus');
    const resultsPanel = document.getElementById('reportResults');
    
    runBtn.disabled = true;
    runBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Investigating...';
    statusPanel.style.display = 'block';
    resultsPanel.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/agent/investigate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const result = await response.json();
        
        statusPanel.style.display = 'none';
        
        if (result.success) {
            resultsPanel.style.display = 'block';
            document.getElementById('reportContent').textContent = result.data.report;
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        statusPanel.style.display = 'none';
        alert('Error: ' + error.message);
    } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = '<i class="fas fa-rocket"></i> RUN FULL INVESTIGATION';
    }
}
