# Notification Skill 使用指南

## 概述

Notification Skill 提供 Windows 桌面通知和 Microsoft Todo 集成功能。

## 功能列表

| 功能 | 命令 | 说明 |
|------|------|------|
| Windows 通知 | `toast:标题\|内容` | 发送 Windows Toast 通知 |
| 定时提醒 | `reminder:标题\|时间` | 设置定时提醒 |
| Microsoft Todo | `todo:add/任务内容` | 添加待办事项 |

## 使用方法

### 1. 基础通知

```
输入: toast:会议通知|15:00有周会
输出: 成功发送 Windows 通知
```

### 2. 定时提醒

```
输入: reminder:喝水|every 2 hours
输出: 已设置周期性提醒
```

### 3. Microsoft Todo

#### 3.1 配置 OAuth

```python
# 在代码中配置
from skills.notification_skill import NotificationSkill

skill = NotificationSkill()
skill.configure_microsoft(
    client_id="your-azure-app-client-id",
    tenant_id="your-tenant-id"
)
# 然后实现 OAuth 认证流程获取 access_token
skill.set_microsoft_token(access_token)
```

#### 3.2 添加待办

```
输入: todo:add:买牛奶
输出: {"success": true, "message": "已添加到 Microsoft Todo"}
```

#### 3.3 列出待办

```
输入: todo:list
输出: {"tasks": [{"title": "买牛奶", "status": "pending"}], "total": 1}
```

#### 3.4 完成任务

```
输入: todo:complete:task_id
输出: {"success": true, "message": "任务已完成"}
```

## 安装依赖

```bash
pip install win10toast
```

## 实现原理

### Windows Toast 通知

使用 `win10toast` 库发送 Windows 10/11 原生通知：

```python
from win10toast import ToastNotifier

toaster = ToastNotifier()
toaster.show_toast(
    title="标题",
    msg="内容",
    duration=5,  # 显示秒数
    threaded=True
)
```

### Microsoft Todo API

使用 Microsoft Graph API：

```
POST https://graph.microsoft.com/v1.0/me/todo/lists/{list_id}/tasks
Authorization: Bearer {access_token}
Content-Type: application/json

{
    "title": "任务标题"
}
```

### OAuth 认证流程

1. 在 Azure Portal 注册应用
2. 配置权限: `Tasks.ReadWrite`
3. 实现 Authorization Code Flow
4. 获取 access_token

## 注意事项

1. **Windows 通知**: 仅支持 Windows 10/11，其他平台会提示安装失败
2. **Microsoft Todo**: 需要有效的 OAuth 认证，未配置时返回友好提示
3. **定时提醒**: 通过 asyncio.sleep 实现，不适用于长期后台任务

## 错误处理

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| `win10toast not installed` | 未安装 | `pip install win10toast` |
| `需要配置 OAuth` | 未设置令牌 | 配置 Microsoft Graph 认证 |
| `时间格式不正确` | 格式错误 | 使用 `HH:MM` 或 `HH:MM YYYY-MM-DD` |
