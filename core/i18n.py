"""Bilingual (English/Chinese) support for the Flavonoid Agent panel ONLY.

Scope is deliberate and narrow: the rest of the app (Formula/Structures/...
tabs, activityBar buttons, leftPanel, core ErrorMsg/WarnMsg/InforMsg flows)
stays English permanently -- see the "GUI env + English-only UI text"
project convention. Only strings built inside the Agent panel (mode switch,
chat UI, Settings dialog, session sidebar, file-viewer button/window) are
looked up through tr().

Language switch takes effect after an app restart (persisted via the same
QtCore.QSettings("AnnoMASS", "AnnoMASS") store already used for the
theme, annomass_app.py's _apply_theme) -- no live re-render is attempted.
"""

LANG_EN = "en"
LANG_ZH = "zh"

TRANSLATIONS = {
    "agent.tab_title": {"en": "Flavonoid Agent", "zh": "黄酮智能体"},
    "agent.mode_label": {"en": "Mode:", "zh": "模式:"},
    "agent.mode_manual": {"en": "Manual", "zh": "手动"},
    "agent.mode_agent": {"en": "Agent", "zh": "智能体"},
    "agent.provider_not_configured": {"en": "Provider: not configured", "zh": "服务商: 未配置"},
    "agent.provider_label": {"en": "Provider: {label} — {model}", "zh": "服务商: {label} — {model}"},
    "agent.settings_button": {"en": "Settings…", "zh": "设置…"},
    "agent.language_label": {"en": "Language:", "zh": "语言:"},
    "agent.new_chat": {"en": "+ New Chat", "zh": "+ 新建对话"},
    "agent.clear_chat": {"en": "Clear Chat", "zh": "清空对话"},
    "agent.clear_chat_tooltip": {
        "en": "Reset the current conversation (keeps it as one session; use +New Chat for a separate one).",
        "zh": "重置当前会话内容(仍是同一个会话;想要独立的新对话请用「+新建对话」)。",
    },
    "agent.send": {"en": "Send", "zh": "发送"},
    "agent.stop": {"en": "Stop", "zh": "停止"},
    "agent.retry": {"en": "Retry", "zh": "重试"},
    "agent.copy_chat": {"en": "Copy Chat", "zh": "复制对话"},
    "agent.export_chat": {"en": "Export", "zh": "导出"},
    "agent.search_sessions": {"en": "Search chats...", "zh": "搜索对话..."},
    "agent.search_files": {"en": "Filter files...", "zh": "筛选文件..."},
    "agent.duplicate_chat": {"en": "Duplicate", "zh": "复制会话"},
    "agent.insert_path": {"en": "Insert path in chat", "zh": "插入路径到输入框"},
    "agent.ask_about_file": {"en": "Ask Agent about this file", "zh": "让 Agent 分析此文件"},
    "agent.use_current_file": {"en": "Use Current File", "zh": "引用当前文件"},
    "agent.input_placeholder": {
        "en": "Ask about results, run the pipeline, look up a compound…",
        "zh": "询问结果、运行流程、查询化合物…",
    },
    "agent.chat_default_name": {"en": "Chat {n}", "zh": "对话 {n}"},
    "agent.rename_chat": {"en": "Rename", "zh": "重命名"},
    "agent.delete_chat": {"en": "Delete Chat", "zh": "删除对话"},
    "agent.delete_chat_confirm": {
        "en": "Delete \"{name}\"? This cannot be undone.",
        "zh": "删除「{name}」？此操作无法撤销。",
    },
    "agent.pin_chat": {"en": "Pin", "zh": "固定"},
    "agent.unpin_chat": {"en": "Unpin", "zh": "取消固定"},
    "agent.sessions_pinned": {"en": "PINNED", "zh": "已固定"},
    "agent.sessions_today": {"en": "TODAY", "zh": "今天"},
    "agent.sessions_yesterday": {"en": "YESTERDAY", "zh": "昨天"},
    "agent.sessions_week": {"en": "PREVIOUS 7 DAYS", "zh": "最近 7 天"},
    "agent.sessions_older": {"en": "OLDER", "zh": "更早"},
    "agent.welcome_title": {"en": "What are you researching today?", "zh": "今天想研究什么?"},
    "agent.welcome_subtitle": {
        "en": "I can help explain identification results, the ranking cascade, or write data-processing code.",
        "zh": "我可以协助你理解鉴定结果、排序逻辑,或者帮你写数据处理代码。",
    },
    "agent.no_provider_configured": {
        "en": "No LLM provider configured. Click Settings to add an API key.",
        "zh": "尚未配置大模型服务商,点击「设置」添加 API Key。",
    },
    "agent.provider_configured": {"en": "Provider configured.", "zh": "服务商已配置。"},
    "agent.thinking": {"en": "Thinking...", "zh": "思考中..."},
    "agent.ready": {"en": "Ready.", "zh": "就绪。"},
    "agent.stopping": {"en": "Stopping after the current request...", "zh": "将在当前请求结束后停止..."},
    "agent.stopped": {"en": "Stopped.", "zh": "已停止。"},
    "agent.chat_copied": {"en": "Chat copied to clipboard.", "zh": "对话已复制到剪贴板。"},
    "agent.chat_exported": {"en": "Chat exported: {path}", "zh": "对话已导出: {path}"},
    "agent.nothing_to_retry": {"en": "No previous user message to retry.", "zh": "没有可重试的上一条用户消息。"},
    "agent.failed_prefix": {"en": "Failed: {message}", "zh": "失败: {message}"},
    "agent.error_prefix": {"en": "Error: {message}", "zh": "错误: {message}"},
    "agent.err_no_provider": {
        "en": "No LLM provider configured yet. Click Settings first.",
        "zh": "还没有配置大模型服务商,请先点击「设置」。",
    },
    "agent.err_still_working": {
        "en": "The agent is still working on the previous message.",
        "zh": "智能体还在处理上一条消息。",
    },
    "agent.err_still_working_session": {
        "en": "The agent is still working on the current conversation. Wait for it to finish first.",
        "zh": "智能体还在处理当前对话,请等它完成后再试。",
    },
    "agent.view_file_button": {"en": "📄 View File…", "zh": "📄 查看文件…"},
    "agent.file_browser_title": {"en": "Files", "zh": "文件"},
    "agent.file_browser_browse": {"en": "Choose folder…", "zh": "选择文件夹…"},
    "agent.file_browser_refresh": {"en": "Refresh", "zh": "刷新"},
    "agent.file_viewer_title": {"en": "File Viewer", "zh": "文件查看器"},
    "agent.file_viewer_open": {"en": "Open…", "zh": "打开…"},
    "agent.file_viewer_prev": {"en": "◀ Prev", "zh": "◀ 上一页"},
    "agent.file_viewer_next": {"en": "Next ▶", "zh": "下一页 ▶"},
    "agent.file_viewer_zoom_in": {"en": "Zoom in", "zh": "放大"},
    "agent.file_viewer_zoom_out": {"en": "Zoom out", "zh": "缩小"},
    "agent.file_viewer_zoom_reset": {"en": "Click to reset to 100%", "zh": "点击恢复到 100%"},
    "agent.file_viewer_page_label": {"en": "Page {current}/{total}", "zh": "第 {current}/{total} 页"},
    "agent.settings_title": {"en": "Flavonoid Agent Settings", "zh": "黄酮智能体设置"},
    "agent.settings_connection": {"en": "API Connection", "zh": "API 连接"},
    "agent.settings_behavior": {"en": "Agent Behavior", "zh": "智能体行为"},
    "agent.settings_provider": {"en": "Provider:", "zh": "服务商:"},
    "agent.settings_protocol": {"en": "Protocol:", "zh": "接口协议:"},
    "agent.settings_protocol_openai": {"en": "OpenAI-compatible tools/chat", "zh": "OpenAI 兼容接口/工具协议"},
    "agent.settings_protocol_anthropic": {"en": "Anthropic Messages/tools", "zh": "Anthropic Messages/工具协议"},
    "agent.settings_base_url": {"en": "Base URL:", "zh": "接口地址:"},
    "agent.settings_model": {"en": "Model:", "zh": "模型:"},
    "agent.settings_api_key": {"en": "API Key:", "zh": "API 密钥:"},
    "agent.settings_show": {"en": "Show", "zh": "显示"},
    "agent.context_cut": {"en": "Cut", "zh": "剪切"},
    "agent.context_copy": {"en": "Copy", "zh": "复制"},
    "agent.context_paste": {"en": "Paste", "zh": "粘贴"},
    "agent.context_clear": {"en": "Clear", "zh": "清空"},
    "agent.context_select_all": {"en": "Select all", "zh": "全选"},
    "agent.settings_test_connection": {"en": "Test Connection", "zh": "测试连接"},
    "agent.settings_thinking_effort": {"en": "Thinking effort:", "zh": "思考强度:"},
    "agent.effort_low": {"en": "Low (fewest tokens)", "zh": "低(省token)"},
    "agent.effort_medium": {"en": "Medium (balanced)", "zh": "中(均衡)"},
    "agent.effort_high": {"en": "High (most thorough)", "zh": "高(最彻底)"},
    "agent.settings_thinking_effort_tooltip": {
        "en": "Low: answer directly, minimal tool calls, fewest tokens. "
              "Medium: balanced (default). High: proactively cross-checks "
              "multiple sources (e.g. MassBank real spectra) for ambiguous "
              "cases like isomers, at the cost of more tokens.",
        "zh": "低:直接作答,尽量少调用工具,消耗token最少。"
              "中:均衡(默认)。"
              "高:遇到同分异构体等有歧义的情况时,主动用多个来源"
              "(如MassBank真实谱图)交叉验证,会消耗更多token。",
    },
    "agent.settings_testing": {"en": "Testing...", "zh": "测试中..."},
    "agent.settings_fill_first": {
        "en": "Fill in Base URL, Model, and API Key first.",
        "zh": "请先填写接口地址、模型和 API 密钥。",
    },
    "agent.settings_connected": {"en": "Connected. ", "zh": "已连接。 "},
    "agent.settings_failed": {"en": "Failed. ", "zh": "失败。 "},
    "agent.restart_required": {
        "en": "Restart the app for the language change to take effect.",
        "zh": "重启应用后语言更改才会生效。",
    },
}


def get_language():
    try:
        from PyQt5 import QtCore
        settings = QtCore.QSettings("AnnoMASS", "AnnoMASS")
        lang = settings.value("language", LANG_EN)
        return lang if lang in (LANG_EN, LANG_ZH) else LANG_EN
    except Exception:
        return LANG_EN


def set_language(lang):
    if lang not in (LANG_EN, LANG_ZH):
        return
    try:
        from PyQt5 import QtCore
        settings = QtCore.QSettings("AnnoMASS", "AnnoMASS")
        settings.setValue("language", lang)
    except Exception:
        pass


def tr(key, **kwargs):
    """Never raises, never shows blank: missing language for a known key
    falls back to English; a totally unknown key falls back to the key
    itself so a typo is visible/debuggable rather than silently empty."""
    entry = TRANSLATIONS.get(key)
    if entry is None:
        text = key
    else:
        lang = get_language()
        text = entry.get(lang) or entry.get(LANG_EN) or key
    if kwargs:
        try:
            return text.format(**kwargs)
        except Exception:
            return text
    return text
