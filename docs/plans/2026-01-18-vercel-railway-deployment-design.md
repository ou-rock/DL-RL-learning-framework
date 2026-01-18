# Vercel + Railway 部署设计文档

**日期**: 2026-01-18
**状态**: 已批准
**作者**: Claude Code + 用户协作

---

## 概述

将 learning-framework-dev 项目部署为完整的 Web 应用：
- **前端**: Vercel（静态托管）
- **后端**: Railway（Python 服务 + PostgreSQL）
- **认证**: 匿名模式（localStorage UUID）

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户浏览器                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  localStorage: { userId: "uuid-xxxx-xxxx" }             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Vercel (前端)                              │
│  learning_framework/visualization/web/                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │index.html│ │ main.css │ │  *.js    │ │ assets   │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │ API 请求 (带 X-User-ID header)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Railway (后端)                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Python HTTP Server (修改版 server.py)                   │   │
│  │  - /api/concepts      获取概念列表                       │   │
│  │  - /api/concept/:id   获取概念详情                       │   │
│  │  - /api/progress      获取/更新用户进度 (新增)            │   │
│  │  - /api/graph         获取知识图谱                       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Railway PostgreSQL (免费实例)                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │   users     │ │  progress   │ │   reviews   │              │
│  │  (user_id)  │ │(concept,lvl)│ │(spaced rep) │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计决策**:
1. 前后端完全分离：Vercel 只托管静态文件，所有 API 请求发往 Railway
2. 无状态后端：用户身份通过 `X-User-ID` header 传递，后端不存储 session
3. CORS 配置：后端需允许 Vercel 域名的跨域请求

---

## 2. 文件变更清单

```
learning-framework-dev/
├── 【新增】Procfile                    # Railway 启动命令
├── 【新增】runtime.txt                 # Python 版本声明
├── 【修改】requirements.txt            # 添加 psycopg2、gunicorn
│
├── learning_framework/
│   ├── 【新增】db/
│   │   ├── __init__.py
│   │   ├── connection.py              # PostgreSQL 连接池
│   │   └── models.py                  # 用户/进度表定义
│   │
│   ├── progress/
│   │   └── 【修改】database.py         # SQLite → PostgreSQL 适配
│   │
│   └── visualization/
│       ├── 【修改】server.py           # 添加 CORS、用户 API
│       └── web/
│           └── js/
│               └── 【修改】api.js      # 添加 userId header
│
├── 【新增】vercel.json                 # Vercel 前端配置
└── 【新增】frontend/                   # 前端独立部署目录（符号链接或复制）
```

### 关键修改说明

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `server.py` | ~50 行 | 添加 CORS headers、新增 `/api/progress` 端点 |
| `database.py` | ~30 行 | 抽象数据库接口，支持 SQLite/PostgreSQL 双模式 |
| `db/connection.py` | ~40 行 | PostgreSQL 连接池，从环境变量读取 `DATABASE_URL` |
| `api.js` | ~15 行 | 生成 userId、添加到请求 header |
| `Procfile` | 1 行 | `web: python -m learning_framework.visualization.server` |

### 环境变量（Railway 配置）

```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname  # Railway 自动提供
PORT=8080                                              # Railway 自动提供
ALLOWED_ORIGINS=https://your-app.vercel.app           # 前端域名
```

---

## 3. 数据库设计

```sql
-- 用户表（匿名用户）
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    user_id UUID UNIQUE NOT NULL,      -- 前端生成的 UUID
    created_at TIMESTAMP DEFAULT NOW()
);

-- 学习进度表
CREATE TABLE progress (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    concept VARCHAR(100) NOT NULL,
    level INTEGER DEFAULT 0,            -- 掌握等级 0-3
    total_score FLOAT DEFAULT 0,
    attempts INTEGER DEFAULT 0,
    last_activity TIMESTAMP,
    UNIQUE(user_id, concept)
);

-- 间隔重复表
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    concept VARCHAR(100) NOT NULL,
    next_review DATE,
    interval_days INTEGER DEFAULT 1,
    ease_factor FLOAT DEFAULT 2.5,      -- SM-2 算法参数
    UNIQUE(user_id, concept)
);
```

---

## 4. 数据流

### 用户首次访问

1. 用户打开 Vercel 前端
2. 前端检查 localStorage，无 userId 则生成 `crypto.randomUUID()`
3. 请求 `/api/concepts`，带 `X-User-ID` header
4. 后端返回概念列表 + 该用户的进度

### 进度保存

1. 用户完成 Quiz
2. 前端 POST `/api/progress` 带分数和时间
3. 后端更新 progress 表，计算下次复习时间
4. 返回成功响应

---

## 5. 错误处理策略

| 场景 | 用户看到的 | 恢复策略 |
|------|-----------|----------|
| 后端冷启动 | "正在唤醒服务器..." | 自动重试 3 次 |
| 网络断开 | "网络连接失败" | 使用 localStorage 缓存 |
| API 500 | "服务器错误" | 显示重试按钮 |
| 进度保存失败 | "已缓存本地" | 存入 localStorage，下次重试 |

### 离线队列

```javascript
class ProgressQueue {
    add(progressData) {
        // 保存到 localStorage
    }
    async flush() {
        // 在线时批量上传
    }
}
```

---

## 6. 部署步骤

### Step 1: Railway 设置

1. 访问 https://railway.app 用 GitHub 登录
2. 新建项目，选择 "Deploy from GitHub repo"
3. 选择 learning-framework-dev 仓库

### Step 2: 创建 PostgreSQL

1. 在 Railway 项目中点击 "New" → "Database" → "PostgreSQL"
2. 等待创建完成
3. DATABASE_URL 会自动注入

### Step 3: 后端配置

添加文件:

**Procfile**:
```
web: python -m learning_framework.visualization.server --port $PORT
```

**runtime.txt**:
```
python-3.11.0
```

### Step 4: Vercel 前端

**vercel.json**:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "learning_framework/visualization/web/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "/learning_framework/visualization/web/$1" }
  ]
}
```

部署:
1. 访问 https://vercel.com 用 GitHub 登录
2. Import Project → 选择仓库
3. 设置 Root Directory: `learning_framework/visualization/web`
4. Deploy

### Step 5: 连接前后端

在 Vercel 环境变量中设置:
```
API_BASE_URL=https://your-backend.railway.app
```

---

## 7. 部署检查清单

- [ ] Railway GitHub 仓库已连接
- [ ] PostgreSQL 已创建并连接
- [ ] 环境变量已配置 (ALLOWED_ORIGINS)
- [ ] 后端 /health 返回 {"status": "ok"}
- [ ] Vercel GitHub 仓库已连接
- [ ] Root Directory 设置正确
- [ ] 前端能获取概念列表
- [ ] 进度能保存到数据库
- [ ] 刷新后进度仍在

---

## 8. 成本预估

| 服务 | 免费额度 | 预计使用 |
|------|----------|----------|
| Railway 计算 | $5/月 | ~$2-3/月（低流量） |
| Railway PostgreSQL | 1GB 存储 | < 100MB |
| Vercel | 无限静态托管 | 免费 |

**总计: 免费**（在免费额度内）

---

## 附录: 为什么选择这个方案

1. **Railway vs Render**: Railway 无冷启动休眠，用户体验更好
2. **PostgreSQL vs SQLite**: 云部署需要持久化存储，SQLite 文件会丢失
3. **UUID 匿名认证**: 最小化实现，快速上线，后续可扩展为登录系统
4. **文件驱动概念数据**: 内容与用户数据分离，便于版本控制
