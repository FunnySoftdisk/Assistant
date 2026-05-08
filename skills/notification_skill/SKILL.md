# Notification Skill 配置

name: notification_skill
version: 1.0.0
description: Windows桌面通知Skill - 发送Windows原生通知，支持提醒事项、待办事项集成
agent_type: execution
priority: 1
tools: ["toast", "reminder", "microsoft_todo"]
parameters: {"title": "通知标题", "message": "通知内容", "duration": 5}
requirements: ["win10toast", "plyer", "pywin32"]

notes: |
    Windows通知实现方案:
    1. 基础方案: win10toast - Windows 10/11 原生Toast通知
    2. 跨平台方案: plyer - 支持多平台通知
    3. Microsoft Todo集成: 需要Azure AD OAuth认证

    使用示例:
    - toast:Windows通知标题:通知内容
    - reminder:开会:15:00开会
    - todo:add:买牛奶