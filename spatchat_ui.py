"""Shared SpatChat room chrome, identical to the Move room's look and feel.

The stylesheet below is the Move room's, copied verbatim; keep this file identical across
the room repos. Inside `with gr.Blocks(...)` a room calls, in order:

    spatchat_ui.build_header(gr, title, subtitle, room)
    with gr.Row(elem_id="spatchat-workarea"):
        with gr.Column(scale=3, min_width=0, elem_id="spatchat-sidebar"):
            chatbot = gr.Chatbot(**spatchat_ui.chatbot_kwargs(gr, welcome))
            status = gr.HTML(value="", elem_id="spatchat-status")
            chips = spatchat_ui.chip_row(gr, ["Prompt one", "Prompt two"])
            upload, textbox, send = spatchat_ui.composer(gr, "Message Spatchat...", [".csv"])
        ...
        with gr.Column(scale=5, min_width=0, elem_id="spatchat-mapcol"):
            with gr.Tabs(elem_id="spatchat-tabs"): ...
    spatchat_ui.build_footer(gr, cite)

and passes `spatchat_ui.HEAD` as part of the page head.
"""
from __future__ import annotations

import base64
import inspect
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def logo_html() -> str:
    try:
        with open(os.path.join(_HERE, "logo_small.png"), "rb") as f:
            src = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
    except OSError:
        return '<span style="font-weight:700;color:#e8527a">Spatchat</span>'
    return f'<img src="{src}" alt="Spatchat" draggable="false">'


HEAD = """
<script>
window.spatchatApplyTheme = function(theme) {
  const dark = theme === "dark";
  document.documentElement.classList.toggle("spatchat-dark", dark);
  document.body.classList.toggle("spatchat-dark", dark);
  document.body.classList.toggle("dark", dark);
  try { localStorage.setItem("spatchat-theme", dark ? "dark" : "light"); } catch (e) {}
};
window.spatchatToggleTheme = function() {
  window.spatchatApplyTheme(document.body.classList.contains("spatchat-dark") ? "light" : "dark");
};
(function initSpatchatTheme() {
  let saved = null;
  try { saved = localStorage.getItem("spatchat-theme"); } catch (e) {}
  const apply = () => {
    window.spatchatApplyTheme(saved === "dark" ? "dark" : "light");
    // Embedded in spatchat.org, which already has its own header.
    let embedded = true;
    try { embedded = window.self !== window.top; } catch (e) {}
    // Served behind the gateway at /<room>/ (not at the site root): show account links.
    if (window.location.pathname.replace(/[/]+$/, "") !== "") document.body.classList.add("spatchat-gated");
    if (embedded) {
      document.body.classList.add("spatchat-embedded");
      document.documentElement.classList.add("spatchat-embedded");
    }
  };
  if (document.body) apply(); else document.addEventListener("DOMContentLoaded", apply, { once: true });
})();
</script>
"""

CSS = """
<style>

    :root {
        --spatchat-connect-bg: #f5f8f5;
        --spatchat-surface: #ffffff;
        --spatchat-surface-alt: #eef3ee;
        --spatchat-border: rgba(16, 36, 26, 0.1);
        --spatchat-text: #10241a;
        --spatchat-text-secondary: #5c6b63;
        --spatchat-text-subtle: #8a978f;
        --spatchat-accent: #1f7a4f;
        --spatchat-accent-hover: #17603d;
        --spatchat-accent-soft: #e3f2e8;
        --spatchat-shadow: 0 1px 3px rgba(16, 36, 26, 0.08), 0 1px 2px rgba(16, 36, 26, 0.06);
    }
    :host,
    html,
    body,
    gradio-app,
    body > gradio-app,
    .gradio-container,
    .gradio-container > .main,
    .gradio-container .main,
    #spatchat-workarea,
    #spatchat-sidebar,
    #spatchat-mapcol,
    #spatchat-splitter {
        background: var(--spatchat-connect-bg) !important;
        color: var(--spatchat-text) !important;
    }
    .gradio-container {
        max-width: 100% !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
        overflow: visible !important;
    }
    #spatchat-workarea {
        overflow-x: auto;
        overflow-y: hidden;
        height: 80vh;
        padding-bottom: 8px;
        box-sizing: border-box;
    }
    #spatchat-workarea > .gradio-row {
        min-width: 900px;
        min-height: 0;
        height: 100%;
        align-items: stretch;
        flex-wrap: nowrap;
        gap: 0 !important;
    }
    #spatchat-sidebar {
        min-width: 0 !important;
        max-width: 1200px;
        flex: 0 0 420px !important;
        width: 420px;
        height: 100%;
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        overflow-y: auto;
        overflow-x: clip;
        padding-right: 4px !important;
        font-size: 15px;
        scrollbar-color: #c9d3cb var(--spatchat-surface-alt);
        scrollbar-width: thin;
    }
    #spatchat-sidebar::-webkit-scrollbar {
        width: 12px;
    }
    #spatchat-sidebar::-webkit-scrollbar-track {
        background: var(--spatchat-surface-alt);
        border-radius: 999px;
    }
    #spatchat-sidebar::-webkit-scrollbar-thumb {
        background: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-surface-alt);
    }
    #spatchat-sidebar::-webkit-scrollbar-thumb:hover {
        background: #b5c1b7;
    }
    #spatchat-sidebar::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-surface-alt);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }
    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-sidebar > div {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    #spatchat-sidebar .form {
        background: transparent !important;
    }
    #spatchat-sidebar label span,
    #spatchat-sidebar .label-wrap span {
        color: var(--spatchat-text-secondary) !important;
    }
    #spatchat-chatbot,
    #spatchat-input-row,
    #spatchat-user-input,
    #spatchat-file-input,
    #spatchat-x-col,
    #spatchat-y-col,
    #spatchat-crs,
    #spatchat-confirm,
    #spatchat-download {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    #spatchat-chatbot {
        padding-left: 0 !important;
        padding-right: 0 !important;
        --chatbot-body-text-size: 16px;
        font-size: 16px !important;
        flex: 1 1 auto !important;
        height: auto !important;
        min-height: 0 !important;
    }
    #spatchat-chatbot > div,
    #spatchat-chatbot .wrap,
    #spatchat-chatbot .bubble-wrap,
    #spatchat-chatbot .message-wrap,
    #spatchat-chatbot .message,
    #spatchat-chatbot .panel-wrap {
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    #spatchat-chatbot .panel-wrap,
    #spatchat-chatbot .wrap,
    #spatchat-chatbot .bubble-wrap,
    #spatchat-chatbot .message-wrap,
    #spatchat-chatbot .message,
    #spatchat-chatbot .message-row.panel,
    #spatchat-chatbot .message-row.panel.user-row,
    #spatchat-chatbot .message-row.panel.bot-row,
    #spatchat-chatbot .wrapper {
        background: var(--spatchat-connect-bg) !important;
        color: var(--spatchat-text) !important;
    }
    #spatchat-chatbot,
    #spatchat-chatbot > div {
        background: var(--spatchat-connect-bg) !important;
        border: none !important;
        box-shadow: none !important;
    }
    #spatchat-chatbot .md,
    #spatchat-chatbot .prose,
    #spatchat-chatbot .md *,
    #spatchat-chatbot .prose * {
        color: var(--spatchat-text) !important;
    }
    #spatchat-chatbot code,
    #spatchat-chatbot pre {
        background: var(--spatchat-surface-alt) !important;
        color: var(--spatchat-text) !important;
    }
    #spatchat-chatbot,
    #spatchat-chatbot > div,
    #spatchat-chatbot .panel-wrap,
    #spatchat-chatbot .wrap {
        scrollbar-color: #c9d3cb var(--spatchat-connect-bg);
        scrollbar-width: thin;
    }
    #spatchat-chatbot::-webkit-scrollbar,
    #spatchat-chatbot > div::-webkit-scrollbar,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar,
    #spatchat-chatbot .wrap::-webkit-scrollbar {
        width: 12px;
    }
    #spatchat-chatbot::-webkit-scrollbar-track,
    #spatchat-chatbot > div::-webkit-scrollbar-track,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-track,
    #spatchat-chatbot .wrap::-webkit-scrollbar-track {
        background: var(--spatchat-connect-bg);
        border-radius: 999px;
    }
    #spatchat-chatbot::-webkit-scrollbar-thumb,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-thumb,
    #spatchat-chatbot .wrap::-webkit-scrollbar-thumb {
        background: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-connect-bg);
    }
    #spatchat-chatbot::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot .wrap::-webkit-scrollbar-thumb:hover {
        background: #b5c1b7;
    }
    #spatchat-chatbot::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot .wrap::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-connect-bg);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }
    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot .wrap::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot .wrap::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-input-row {
        align-items: center !important;
        gap: 8px !important;
        flex-wrap: nowrap !important;
        background: var(--spatchat-surface) !important;
        border: 1px solid var(--spatchat-border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
        padding: 6px 8px !important;
        margin-top: -8px !important;
    }
    #spatchat-input-row .form {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    #spatchat-chatbot {
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
        overflow: hidden !important;
    }
    #spatchat-chatbot > div {
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    }
    #spatchat-file-input {
        flex: 0 0 34px !important;
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: none !important;
        background: var(--spatchat-accent) !important;
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }
    #spatchat-file-input:hover {
        background: var(--spatchat-accent-hover) !important;
    }
    #spatchat-file-input:active {
        transform: scale(0.92) !important;
    }
    #spatchat-sample-btn {
        align-self: flex-start !important;
        margin-top: 6px !important;
        padding: 4px 10px !important;
        min-width: 0 !important;
        background: transparent !important;
        color: var(--spatchat-accent) !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        transition: background 0.15s ease !important;
    }
    #spatchat-sample-btn:hover {
        background: var(--spatchat-accent-soft) !important;
        border-radius: var(--radius-md) !important;
    }
    #spatchat-user-input,
    #spatchat-user-input > div,
    #spatchat-user-input .wrap,
    #spatchat-user-input label {
        background: transparent !important;
    }
    #spatchat-user-input {
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }
    #spatchat-user-input .input-container {
        align-items: center !important;
    }
    #spatchat-user-input textarea,
    #spatchat-user-input input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 34px !important;
        box-sizing: border-box !important;
        padding: 8px 4px !important;
        line-height: 20px !important;
        color: var(--spatchat-text) !important;
    }
    #spatchat-user-input textarea {
        max-height: 200px !important;
        overflow-y: auto !important;
        resize: vertical !important;
    }
    #spatchat-user-input input {
        overflow: hidden !important;
    }
    #spatchat-user-input textarea::placeholder,
    #spatchat-user-input input::placeholder {
        color: var(--spatchat-text-subtle) !important;
        text-align: left !important;
    }
    #spatchat-chatbot .message-row.panel,
    #spatchat-chatbot .message-row.panel.user-row,
    #spatchat-chatbot .message-row.panel.bot-row {
        background: transparent !important;
        margin: 0 !important;
        padding: 6px 0 !important;
    }
    #spatchat-chatbot .message-row.panel.user-row,
    #spatchat-chatbot .message-row.panel.bot-row {
        background: transparent !important;
    }
    #spatchat-chatbot .message {
        width: 100% !important;
    }
    #spatchat-chatbot .flex-wrap.user,
    #spatchat-chatbot .flex-wrap.bot {
        border: none !important;
        box-shadow: none !important;
        padding: 8px 12px !important;
    }
    #spatchat-chatbot .flex-wrap.bot {
        background: transparent !important;
        border-bottom-left-radius: var(--radius-md) !important;
    }
    #spatchat-chatbot .flex-wrap.user {
        background: var(--spatchat-accent-soft) !important;
        border-bottom-right-radius: var(--radius-md) !important;
    }
    #spatchat-chatbot .message-row.panel.user-row {
        align-self: stretch !important;
        width: 100% !important;
        background: transparent !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .message {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
        margin-bottom: 0 !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .message > div {
        display: block !important;
        width: max-content !important;
        max-width: min(72%, 680px) !important;
        height: auto !important;
        flex: 0 1 auto !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        align-self: flex-end !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user {
        display: inline-block !important;
        vertical-align: top !important;
        width: max-content !important;
        max-width: 100% !important;
        height: auto !important;
        flex: 0 1 auto !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        align-self: flex-end !important;
        background: var(--spatchat-accent-soft) !important;
        color: var(--spatchat-text) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-md) !important;
        border-bottom-right-radius: 0 !important;
        padding: 8px 12px !important;
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        text-align: left !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user {
        display: block !important;
        width: max-content !important;
        max-width: 100% !important;
        min-width: max-content !important;
        height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user *,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user * {
        text-align: left !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user > div,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user > div,
    #spatchat-chatbot .message-row.panel.user-row .user > div > div,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user > div > div,
    #spatchat-chatbot .message-row.panel.user-row .user .wrapper,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .wrapper,
    #spatchat-chatbot .message-row.panel.user-row .user .md,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .md,
    #spatchat-chatbot .message-row.panel.user-row .user .prose,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .prose,
    #spatchat-chatbot .message-row.panel.user-row .user p,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user p {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        width: auto !important;
        max-width: 100% !important;
        min-width: 0 !important;
        height: auto !important;
        min-height: 0 !important;
        flex-grow: 0 !important;
        gap: 0 !important;
        margin: 0 !important;
        white-space: normal !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user .md pre,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .md pre {
        white-space: pre !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user .md > *,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .md > *,
    #spatchat-chatbot .message-row.panel.user-row .user .prose > *,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .prose > * {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user div:empty,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user div:empty,
    #spatchat-chatbot .message-row.panel.user-row .user p:empty,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user p:empty {
        display: none !important;
    }
    #spatchat-chatbot .spatchat-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--spatchat-text-secondary);
        font-weight: 500;
    }
    #spatchat-chatbot .spatchat-status-time {
        color: var(--spatchat-text-secondary);
        font-size: 0.92em;
    }
    #spatchat-status {
        min-height: 24px;
        margin: 4px 0 8px 0 !important;
        color: var(--spatchat-text-secondary);
        font-size: 14px;
    }
    #spatchat-status .spatchat-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--spatchat-text-secondary);
        font-weight: 500;
    }
    #spatchat-status .spatchat-status-time {
        color: var(--spatchat-text-secondary);
        font-size: 0.92em;
    }
    .spatchat-typing-dots {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .spatchat-typing-dots span {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: var(--spatchat-accent);
        animation: spatchat-typing-bounce 1.2s ease-in-out infinite;
    }
    .spatchat-typing-dots span:nth-child(2) {
        animation-delay: 0.15s;
    }
    .spatchat-typing-dots span:nth-child(3) {
        animation-delay: 0.3s;
    }
    @keyframes spatchat-typing-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-3px); opacity: 1; }
    }
    #spatchat-splitter {
        min-width: 10px;
        max-width: 10px;
        width: 10px;
        flex: 0 0 10px !important;
        position: relative;
        padding: 0 !important;
        margin: 0 !important;
        height: 100%;
        min-height: 0;
        align-self: stretch !important;
        background: transparent !important;
        overflow: visible !important;
    }
    #spatchat-splitter > div,
    #spatchat-splitter .gradio-html,
    #spatchat-splitter .gradio-html > div {
        height: 100%;
        min-height: 0;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }
    #spatchat-splitter .spatchat-splitter-handle {
        position: absolute;
        inset: 0;
        display: block;
        width: 100%;
        height: 100%;
        cursor: col-resize;
        user-select: none;
        background: transparent;
    }
    #spatchat-splitter .spatchat-splitter-handle::before {
        content: "";
        position: absolute;
        top: 8px;
        bottom: 8px;
        left: 50%;
        width: 1px;
        transform: translateX(-50%);
        background: var(--spatchat-border);
        border-radius: 999px;
        transition: background 120ms ease, width 120ms ease;
    }
    #spatchat-splitter:hover .spatchat-splitter-handle::before,
    body.spatchat-resizing #spatchat-splitter .spatchat-splitter-handle::before {
        width: 2px;
        background: var(--spatchat-accent);
    }
    body.spatchat-resizing,
    body.spatchat-resizing * {
        cursor: col-resize !important;
        user-select: none !important;
    }
    #spatchat-mapcol {
        min-width: 0 !important;
        flex: 1 1 auto !important;
        width: auto !important;
        height: 100%;
        position: relative;
        overflow: visible;
    }
    #spatchat-mapcol.spatchat-map-dragover {
        outline: 3px dashed var(--spatchat-accent);
        outline-offset: -3px;
        border-radius: var(--radius-lg);
    }
    #spatchat-mapcol.spatchat-map-dragover::after {
        content: "Drop CSV to upload";
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(31, 122, 79, 0.14);
        color: var(--spatchat-accent-hover);
        font-size: 13px;
        font-weight: 600;
        pointer-events: none;
        z-index: 500;
        border-radius: var(--radius-lg);
    }
    #spatchat-map {
        min-height: 0;
        height: 100%;
        overflow: visible;
        width: 100%;
        padding-bottom: 10px;
        box-sizing: border-box;
    }
    #spatchat-map > div {
        height: 100%;
    }
    #spatchat-map iframe {
        width: 100% !important;
        height: 100% !important;
        min-height: 0;
        border: none;
    }
    body.spatchat-resizing #spatchat-map iframe {
        pointer-events: none !important;
    }
    #spatchat-figure-layer {
        position: relative;
        width: 0;
        height: 0;
        z-index: 2000;
        background: transparent !important;
        pointer-events: auto;
        overflow: visible !important;
    }
    #spatchat-figure-layer > div,
    #spatchat-figure-layer .spatchat-figure-root {
        width: 0;
        height: 0;
        background: transparent !important;
        overflow: visible !important;
    }
    .spatchat-figure-root {
        position: relative;
        pointer-events: auto;
    }
    .spatchat-figure-launcher {
        position: fixed;
        right: 22px;
        bottom: 22px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 11px 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        background: rgba(9, 13, 21, 0.96);
        color: #f4f7fb;
        pointer-events: auto;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.38);
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
    }
    .spatchat-figure-modal-backdrop {
        position: fixed;
        inset: 0;
        background: transparent;
        backdrop-filter: none;
        pointer-events: none;
    }
    .spatchat-figure-interaction-shield {
        position: fixed;
        inset: 0;
        background: transparent;
        pointer-events: none;
        z-index: 1999;
    }
    .spatchat-figure-interaction-shield.is-active {
        pointer-events: auto;
        cursor: inherit;
    }
    .spatchat-figure-modal-backdrop.is-hidden {
        display: none;
    }
    .spatchat-figure-modal {
        position: fixed;
        left: clamp(12px, calc(50vw - 280px), calc(100vw - 572px));
        top: clamp(12px, calc(50vh - 230px), calc(100vh - 472px));
        width: min(560px, calc(100vw - 24px));
        height: min(460px, calc(100vh - 24px));
        max-width: calc(100vw - 24px);
        max-height: calc(100vh - 24px);
        margin: 0;
        display: flex;
        flex-direction: column;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(9, 13, 21, 0.97);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.46);
        color: #f4f7fb;
        overflow: hidden;
        pointer-events: auto;
        transform: none;
        z-index: 2001;
    }
    .spatchat-figure-modal.is-hidden {
        display: none;
    }
    .spatchat-figure-modal-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 6px 10px;
        background: linear-gradient(135deg, rgba(36, 51, 82, 0.94), rgba(11, 15, 25, 0.94));
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        cursor: move;
        user-select: none;
    }
    .spatchat-figure-modal-copy {
        display: flex;
        flex-direction: column;
        gap: 0;
        min-width: 0;
    }
    .spatchat-figure-modal-title {
        font-size: 12px;
        font-weight: 700;
        line-height: 1.15;
    }
    .spatchat-figure-modal-count {
        font-size: 10px;
        line-height: 1.1;
        color: rgba(226, 233, 245, 0.74);
    }
    .spatchat-figure-modal-nav {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .spatchat-figure-modal-btn {
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: none;
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.08);
        color: #f4f7fb;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
    }
    .spatchat-figure-modal-btn:disabled {
        opacity: 0.35;
        cursor: default;
    }
    .spatchat-figure-modal-btn-icon svg {
        width: 14px;
        height: 14px;
        stroke: currentColor;
        stroke-width: 1.8;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .spatchat-figure-modal-btn-close {
        font-size: 16px;
        line-height: 0.7;
        padding-bottom: 6px;
    }
    .spatchat-figure-modal-body {
        flex: 1 1 auto;
        overflow: auto;
        padding: 18px;
        display: flex;
        flex-direction: column;
    }
    .spatchat-figure-resize-handle {
        position: absolute;
        z-index: 3;
        background: transparent;
    }
    .spatchat-figure-resize-handle.is-n {
        left: 12px;
        right: 12px;
        top: -4px;
        height: 8px;
        cursor: ns-resize;
    }
    .spatchat-figure-resize-handle.is-s {
        left: 12px;
        right: 12px;
        bottom: -4px;
        height: 8px;
        cursor: ns-resize;
    }
    .spatchat-figure-resize-handle.is-e {
        top: 12px;
        bottom: 12px;
        right: -4px;
        width: 8px;
        cursor: ew-resize;
    }
    .spatchat-figure-resize-handle.is-w {
        top: 12px;
        bottom: 12px;
        left: -4px;
        width: 8px;
        cursor: ew-resize;
    }
    .spatchat-figure-resize-handle.is-ne,
    .spatchat-figure-resize-handle.is-nw,
    .spatchat-figure-resize-handle.is-se,
    .spatchat-figure-resize-handle.is-sw {
        width: 18px;
        height: 18px;
    }
    .spatchat-figure-resize-handle.is-ne {
        top: -4px;
        right: -4px;
        cursor: nesw-resize;
    }
    .spatchat-figure-resize-handle.is-nw {
        top: -4px;
        left: -4px;
        cursor: nwse-resize;
    }
    .spatchat-figure-resize-handle.is-se {
        right: -4px;
        bottom: -4px;
        cursor: nwse-resize;
    }
    .spatchat-figure-resize-handle.is-sw {
        left: -4px;
        bottom: -4px;
        cursor: nesw-resize;
    }
    .spatchat-figure-resize-handle.is-se::before {
        content: "";
        position: absolute;
        right: 7px;
        bottom: 7px;
        width: 11px;
        height: 11px;
        border-right: 2px solid rgba(244, 247, 251, 0.72);
        border-bottom: 2px solid rgba(244, 247, 251, 0.72);
        border-bottom-right-radius: 2px;
    }
    .spatchat-figure-card {
        display: flex;
        flex-direction: column;
        flex: 1 1 auto;
        gap: 10px;
        min-height: 0;
        padding: 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .spatchat-figure-card-title {
        font-size: 16px;
        font-weight: 700;
        line-height: 1.25;
    }
    .spatchat-figure-card-meta {
        font-size: 13px;
        color: rgba(226, 233, 245, 0.74);
        line-height: 1.35;
    }
    .spatchat-figure-card-image {
        width: 100%;
        flex: 1 1 auto;
        min-height: 0;
        height: 100%;
        max-height: none;
        object-fit: contain;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
    }
    .spatchat-figure-card-table {
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
    }
    .spatchat-figure-table-wrap {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .spatchat-figure-table-note {
        font-size: 11px;
        color: rgba(226, 233, 245, 0.72);
    }
    .spatchat-figure-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .spatchat-figure-table th,
    .spatchat-figure-table td {
        padding: 7px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        text-align: left;
        white-space: nowrap;
    }
    .spatchat-figure-table th {
        position: sticky;
        top: 0;
        background: rgba(13, 18, 29, 0.98);
        z-index: 1;
        font-weight: 700;
    }
    .spatchat-figure-table-empty {
        font-size: 12px;
        color: rgba(226, 233, 245, 0.72);
    }
    #spatchat-x-col,
    #spatchat-y-col,
    #spatchat-crs {
        background: var(--spatchat-surface) !important;
        border: 1px solid var(--spatchat-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 8px 10px !important;
        margin-top: 8px !important;
        box-shadow: var(--spatchat-shadow);
    }
    #spatchat-confirm {
        margin-top: 8px !important;
        background: var(--spatchat-accent) !important;
        color: #ffffff !important;
        border: none !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }
    #spatchat-confirm:hover {
        background: var(--spatchat-accent-hover) !important;
    }
    #spatchat-confirm:active {
        transform: scale(0.98) !important;
    }
    #spatchat-download {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 10px;
    }
    #spatchat-download button {
        min-width: 220px;
        width: 100%;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px;
        text-align: center;
        background: var(--spatchat-accent) !important;
        color: #ffffff !important;
        border: none !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }
    #spatchat-download button:hover {
        background: var(--spatchat-accent-hover) !important;
    }
    #spatchat-download button:active {
        transform: scale(0.98) !important;
    }
    footer {
        display: none !important;
    }
    #spatchat-header-row {
        align-items: center !important;
        justify-content: space-between !important;
        flex-wrap: nowrap !important;
        gap: 16px !important;
        background: var(--spatchat-accent-soft) !important;
        border-radius: var(--radius-lg, 12px) !important;
        padding: 6px 18px !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
    }
    #logo-img,
    #logo-img > div,
    #logo-img button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        justify-content: flex-start !important;
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }
    #logo-img img {
        height: 32px;
        margin: 6px 0;
        border-radius: 6px;
    }
    #spatchat-navbar {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }
    body.spatchat-resizing,
    body.spatchat-resizing * {
        cursor: col-resize !important;
        user-select: none !important;
    }
    body.spatchat-figure-moving,
    body.spatchat-figure-moving * {
        cursor: move !important;
        user-select: none !important;
    }
    body.spatchat-figure-resizing,
    body.spatchat-figure-resizing * {
        user-select: none !important;
    }
    @media (max-width: 900px) {
        .spatchat-figure-launcher {
            right: 14px;
            bottom: 14px;
        }
        .spatchat-figure-modal {
            left: 12px;
            top: 12px;
            width: calc(100vw - 24px);
            height: calc(100vh - 24px);
            max-height: calc(100vh - 24px);
        }
        .spatchat-figure-modal-head {
            padding: 14px;
            align-items: flex-start;
            flex-direction: column;
        }
        .spatchat-figure-modal-nav {
            width: 100%;
            justify-content: flex-end;
        }
        .spatchat-figure-modal-body {
            padding: 14px;
        }
        .spatchat-figure-card-image {
            height: 100%;
        }
    }
    #spatchat-share-btn {
        transition: background 0.15s ease, transform 0.1s ease;
    }
    #spatchat-share-btn:hover {
        background: #17603d !important;
    }
    #spatchat-share-btn:active {
        transform: scale(0.96);
    }
    .spatchat-map-empty-hint {
        position: absolute !important;
        inset: 0 !important;
        display: flex !important;
        align-items: flex-end !important;
        justify-content: center !important;
        padding-bottom: 8% !important;
        pointer-events: none !important;
        z-index: 500 !important;
    }
    .spatchat-map-empty-hint-card {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 4px !important;
        padding: 8px 14px !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        border: 1px solid rgba(16, 36, 26, 0.12) !important;
        color: #10241a !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-align: center !important;
        box-shadow: 0 8px 24px rgba(16, 36, 26, 0.16) !important;
    }
    .spatchat-map-empty-hint-card * {
        color: inherit !important;
    }
    .spatchat-map-empty-hint-icon {
        font-size: 16px !important;
    }
    .spatchat-map-empty-hint-sub {
        font-size: 10px !important;
        font-weight: 500 !important;
        color: #5c6b63 !important;
    }
    

    /* ---- Tabs (Map / Plots / Datasheet) ---- */
    #spatchat-mapcol { flex: 1 1 0 !important; width: 0 !important; min-width: 0 !important; overflow: hidden !important; }
    #spatchat-tabs { height: 100%; width: 100%; display: flex; flex-direction: column; background: transparent !important; }
    #spatchat-tabs > .tab-wrapper, #spatchat-tabs > .tab-container { flex: 0 0 auto; }
    #spatchat-tabs > div[role="tabpanel"], #spatchat-tabs .tabitem {
        flex: 1 1 auto; min-height: 0; height: calc(100% - 44px); overflow: auto;
        background: var(--spatchat-surface) !important; border-radius: 0 0 var(--radius-lg) var(--radius-lg);
        padding: 8px !important;
    }
    #spatchat-tabs button[role="tab"].selected { color: var(--spatchat-accent) !important; border-color: var(--spatchat-accent) !important; }
    #spatchat-datasheet { height: 100%; }
    #spatchat-tabs .tabitem { padding: 0 !important; margin-top: 4px; }
    #spatchat-tabs .tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane) { background: transparent !important; overflow: hidden !important; }
    #spatchat-tabs .tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane) > .column,
    #spatchat-tabs .tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane) #spatchat-map { height: 100% !important; width: 100% !important; padding-bottom: 0 !important; }
    #spatchat-tabs div.tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane):not(#x) { padding: 0 !important; }
    #spatchat-tabs #spatchat-map .html-container { padding: 0 !important; height: 100% !important; }
    #spatchat-tabs #spatchat-map .prose, #spatchat-tabs #spatchat-map .prose > div:not(.spatchat-map-empty-hint), #spatchat-tabs #spatchat-map .prose > div:not(.spatchat-map-empty-hint) > div,
    #spatchat-tabs #spatchat-map iframe { height: 100% !important; width: 100% !important; }
    #spatchat-plots-pane, #spatchat-data-pane { padding: 8px !important; }
    /* ---- Plots tab: dock the figure viewer instead of a floating window ---- */
    #spatchat-figure-layer, #spatchat-figure-layer > div, #spatchat-figure-layer .spatchat-figure-root {
        width: 100% !important; height: 100% !important; overflow: visible !important;
    }
    #spatchat-figure-layer .spatchat-figure-launcher,
    #spatchat-figure-layer .spatchat-figure-modal-backdrop,
    #spatchat-figure-layer .spatchat-figure-resize-handle,
    #spatchat-figure-layer .spatchat-figure-modal-btn-close { display: none !important; }
    #spatchat-figure-layer .spatchat-figure-modal {
        position: static !important; display: flex !important; width: 100% !important; height: 100% !important;
        max-width: none !important; max-height: none !important; box-shadow: none !important;
        border-radius: var(--radius-lg) !important;
    }
    #spatchat-figure-layer .spatchat-figure-modal-head { cursor: default !important; }
    #spatchat-figure-layer .spatchat-figure-card-image { max-width: 100% !important; height: auto !important; }
    #spatchat-tabs, #spatchat-tabs .tabitem, #spatchat-plots-pane, #spatchat-plots-pane * { min-width: 0 !important; }
    #spatchat-tabs, #spatchat-tabs .tabitem, #spatchat-plots-pane, #spatchat-plots-pane * { max-width: 100% !important; }
    #spatchat-tabs .tabitem { overflow: auto !important; align-items: flex-start; }
    #spatchat-figure-layer, #spatchat-figure-layer > div, #spatchat-figure-layer .spatchat-figure-root,
    #spatchat-figure-layer .spatchat-figure-modal { height: auto !important; }
    #spatchat-figure-layer .spatchat-figure-modal-body { overflow: auto; }
    #spatchat-figure-layer .spatchat-figure-modal { background: var(--spatchat-surface-alt) !important; border: 1px solid var(--spatchat-border) !important; color: var(--spatchat-text) !important; }
    #spatchat-figure-layer .spatchat-figure-modal-head { background: var(--spatchat-accent-soft) !important; border-bottom: 1px solid var(--spatchat-border) !important; }
    #spatchat-figure-layer .spatchat-figure-modal-title, #spatchat-figure-layer .spatchat-figure-modal-count,
    #spatchat-figure-layer .spatchat-figure-card-title, #spatchat-figure-layer .spatchat-figure-card-meta { color: var(--spatchat-text) !important; }
    #spatchat-figure-layer .spatchat-figure-card { background: var(--spatchat-surface) !important; }
    /* ---- Professional polish ---- */
    :root { --spatchat-font: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; }
    html, body, gradio-app, .gradio-container, .gradio-container * { font-family: var(--spatchat-font) !important; }
    .gradio-container code, .gradio-container pre, .gradio-container pre * { font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important; }
    #spatchat-header-row { background: var(--spatchat-surface) !important; border: 1px solid var(--spatchat-border) !important;
        border-radius: 14px !important; padding: 6px 18px !important; box-shadow: var(--spatchat-shadow); align-items: center !important; }
    #spatchat-navbar a { font-weight: 500 !important; font-size: 14px !important; padding: 6px 2px; border-bottom: 2px solid transparent; transition: border-color .15s, color .15s; }
    #spatchat-navbar a:hover { border-bottom-color: var(--spatchat-accent); color: var(--spatchat-accent) !important; }
    .spatchat-title-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin: 22px 4px 14px; }
    .spatchat-title { margin: 0 !important; font-size: 26px !important; font-weight: 700 !important; letter-spacing: -0.02em; color: var(--spatchat-text) !important; line-height: 1.2; }
    .spatchat-subtitle { margin: 4px 0 0 !important; font-size: 14px !important; color: var(--spatchat-text-secondary) !important; }
    #spatchat-share-btn { flex: 0 0 auto; padding: 8px 18px; font-size: 13px; font-weight: 600; background: transparent; color: var(--spatchat-accent);
        border: 1px solid var(--spatchat-accent); border-radius: 8px; cursor: pointer; transition: background .15s, color .15s; }
    #spatchat-share-btn:hover { background: var(--spatchat-accent); color: #fff; }
    #spatchat-tabs button[role="tab"] { font-size: 14px !important; font-weight: 600 !important; padding: 8px 18px !important; color: var(--spatchat-text-secondary) !important; }
    #spatchat-tabs button[role="tab"].selected { color: var(--spatchat-accent) !important; }
    #spatchat-tabs .tabitem { border: 1px solid var(--spatchat-border) !important; box-shadow: var(--spatchat-shadow); border-radius: 12px !important; }
    #spatchat-chatbot { box-shadow: var(--spatchat-shadow) !important; background: var(--spatchat-surface) !important; }
    #spatchat-chatbot .panel-wrap, #spatchat-chatbot .wrap, #spatchat-chatbot .bubble-wrap, #spatchat-chatbot .message-wrap { background: var(--spatchat-surface) !important; }
    #spatchat-input-row { background: var(--spatchat-surface) !important; box-shadow: var(--spatchat-shadow); }
    #spatchat-input-row:focus-within { border-color: var(--spatchat-accent) !important; box-shadow: 0 0 0 3px var(--spatchat-accent-soft); }
    #spatchat-sample-btn { border: 1px solid var(--spatchat-border) !important; border-radius: 8px !important; background: var(--spatchat-surface) !important;
        align-self: stretch !important; padding: 8px 12px !important; font-size: 13px !important; }
    #spatchat-download, #spatchat-confirm { border-radius: 8px !important; font-weight: 600 !important; }
    #spatchat-download { background: var(--spatchat-accent) !important; color: #fff !important; border: none !important; }
    #spatchat-download:hover { background: var(--spatchat-accent-hover) !important; }
    #spatchat-datasheet table { font-size: 13px; }
    #spatchat-datasheet th { background: var(--spatchat-surface-alt) !important; font-weight: 600 !important; }
    /* ---- Round chatbot ---- */
    #spatchat-chatbot, #spatchat-chatbot > div { border-radius: 24px !important; }
    #spatchat-chatbot { border: 1px solid var(--spatchat-border) !important; }
    #spatchat-input-row {
        border: 1px solid var(--spatchat-border) !important; border-radius: 999px !important;
        margin-top: 8px !important; padding: 4px 10px !important;
    }

    /* ================= ChatGPT-style chat ================= */
    #spatchat-sidebar { gap: 0 !important; flex: 0 0 440px !important; width: 440px; }
    #spatchat-chatbot, #spatchat-chatbot > div, #spatchat-chatbot .wrapper, #spatchat-chatbot .panel-wrap,
    #spatchat-chatbot .bubble-wrap, #spatchat-chatbot .message-wrap {
        background: transparent !important; border: none !important; box-shadow: none !important; border-radius: 0 !important;
    }
    #spatchat-chatbot { flex: 1 1 auto !important; height: auto !important; min-height: 0 !important; padding: 0 !important; }
    #spatchat-chatbot .message-wrap { padding: 4px 4px 12px !important; gap: 18px !important; }
    #spatchat-chatbot .message-row { margin: 0 !important; padding: 0 !important; }
    #spatchat-chatbot .message, #spatchat-chatbot .message * { font-size: 15px; line-height: 1.65; }
    #spatchat-chatbot .message { max-width: 100% !important; padding: 0 !important; border: none !important; box-shadow: none !important; }
    /* assistant: plain text on the page, no box */
    #spatchat-chatbot .bot, #spatchat-chatbot .bot.message, #spatchat-chatbot .message.bot {
        background: transparent !important; border: none !important; padding: 0 !important; width: 100% !important;
    }
    /* user: soft right-aligned bubble */
    #spatchat-chatbot .message-row.user-row, #spatchat-chatbot .user-row { justify-content: flex-end !important; }
    #spatchat-chatbot .user, #spatchat-chatbot .user.message, #spatchat-chatbot .message.user {
        background: var(--spatchat-surface-alt) !important; color: var(--spatchat-text) !important;
        border-radius: 20px 20px 6px 20px !important; padding: 10px 16px !important; width: fit-content !important; max-width: 85% !important;
        margin-left: auto !important;
    }
    #spatchat-chatbot .message p { margin: 0 0 10px !important; }
    #spatchat-chatbot .message p:last-child { margin-bottom: 0 !important; }
    #spatchat-chatbot .message ul, #spatchat-chatbot .message ol { margin: 6px 0 10px 4px !important; padding-left: 20px !important; }
    #spatchat-chatbot .message li { margin: 3px 0 !important; }
    #spatchat-chatbot .message code { padding: 1px 6px !important; border-radius: 6px !important; font-size: 13px !important; }
    #spatchat-chatbot .message pre { border-radius: 10px !important; padding: 12px 14px !important; border: 1px solid var(--spatchat-border) !important; }
    #spatchat-chatbot .icon-button-wrapper { opacity: 0.6; }
    #spatchat-status { min-height: 0 !important; padding: 0 6px !important; color: var(--spatchat-text-secondary) !important; font-size: 13px !important; }

    /* suggestion chips */
    #spatchat-chips { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; padding: 8px 2px 10px !important; background: transparent !important; border: none !important; }
    #spatchat-chips > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    .spatchat-chip, #spatchat-sample-btn.spatchat-chip {
        width: auto !important; min-width: 0 !important; flex: 0 0 auto !important; padding: 6px 14px !important; margin: 0 !important;
        font-size: 13px !important; font-weight: 500 !important; line-height: 1.3 !important; border-radius: 999px !important;
        background: var(--spatchat-surface) !important; color: var(--spatchat-text) !important; border: 1px solid var(--spatchat-border) !important;
        box-shadow: none !important; align-self: auto !important; transition: background .15s, border-color .15s, color .15s;
    }
    .spatchat-chip:hover { background: var(--spatchat-accent-soft) !important; border-color: var(--spatchat-accent) !important; color: var(--spatchat-accent) !important; }

    /* composer */
    #spatchat-input-row {
        background: var(--spatchat-surface) !important; border: 1px solid var(--spatchat-border) !important; border-radius: 28px !important;
        padding: 6px 8px 6px 8px !important; margin: 0 0 10px !important; box-shadow: 0 2px 10px rgba(16,36,26,.10) !important;
        min-height: 54px; align-items: center !important;
    }
    #spatchat-input-row:focus-within { border-color: var(--spatchat-accent) !important; box-shadow: 0 0 0 3px var(--spatchat-accent-soft), 0 2px 10px rgba(16,36,26,.10) !important; }
    #spatchat-user-input textarea, #spatchat-user-input input { font-size: 15px !important; padding: 10px 6px !important; }
    #spatchat-file-input { width: 36px !important; height: 36px !important; min-width: 36px !important; flex: 0 0 36px !important; background: var(--spatchat-surface-alt) !important; color: var(--spatchat-text) !important; font-size: 22px !important; }
    #spatchat-file-input:hover { background: var(--spatchat-accent-soft) !important; color: var(--spatchat-accent) !important; }
    #spatchat-send-btn {
        flex: 0 0 38px !important; width: 38px !important; min-width: 38px !important; height: 38px !important; padding: 0 !important; margin: 0 !important;
        border-radius: 50% !important; border: none !important; background: var(--spatchat-accent) !important; color: #fff !important;
        font-size: 18px !important; font-weight: 700 !important; line-height: 1 !important; display: flex; align-items: center; justify-content: center;
    }
    #spatchat-send-btn:hover { background: var(--spatchat-accent-hover) !important; }

    /* ================= titles / header actions ================= */
    .spatchat-title-actions { display: flex; gap: 8px; align-items: center; }
    #spatchat-theme-btn { padding: 8px 16px; font-size: 13px; font-weight: 600; background: transparent; color: var(--spatchat-text-secondary);
        border: 1px solid var(--spatchat-border); border-radius: 8px; cursor: pointer; transition: background .15s, color .15s; }
    #spatchat-theme-btn:hover { background: var(--spatchat-surface-alt); color: var(--spatchat-text); }
    .spatchat-theme-dark { display: none; }
    body.spatchat-dark .spatchat-theme-light { display: none; }
    body.spatchat-dark .spatchat-theme-dark { display: inline; }

    /* ================= dataset selector ================= */
    #spatchat-dataset-select { flex: 0 0 auto !important; margin-bottom: 8px; background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
    #spatchat-dataset-select label > span, #spatchat-dataset-select .label-wrap span { font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .04em; color: var(--spatchat-text-secondary) !important; }
    #spatchat-dataset-select input { font-size: 14px !important; font-weight: 600 !important; }
    #spatchat-mapcol { flex-wrap: nowrap !important; flex-direction: column !important; }
    #spatchat-tabs { flex: 1 1 0 !important; min-height: 0 !important; }
    #spatchat-dataset-select { width: 100% !important; min-width: 0 !important; }
    #spatchat-dataset-select .wrap, #spatchat-dataset-select input { background: var(--spatchat-surface) !important; border-radius: 10px !important; }
    #spatchat-dataset-select input { border: 1px solid var(--spatchat-border) !important; padding: 8px 12px !important; }

    /* ================= empty states ================= */
    .spatchat-empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
        gap: 8px; min-height: 360px; padding: 32px 24px; color: var(--spatchat-text-subtle); }
    .spatchat-empty-state svg { color: var(--spatchat-text-subtle); opacity: .8; }
    .spatchat-empty-title { font-size: 16px; font-weight: 600; color: var(--spatchat-text); }
    .spatchat-empty-text { font-size: 14px; max-width: 360px; line-height: 1.55; color: var(--spatchat-text-secondary); }


    /* ================= plot gallery + lightbox ================= */
    .spatchat-gallery { display: flex; flex-direction: column; gap: 10px; width: 100%; }
    .spatchat-gallery-bar { display: flex; justify-content: space-between; align-items: baseline; font-size: 13px; font-weight: 600; color: var(--spatchat-text); padding: 2px 2px 0; }
    .spatchat-gallery-hint { font-weight: 400; color: var(--spatchat-text-secondary); font-size: 12px; }
    .spatchat-gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
    .spatchat-thumb { position: relative; display: flex !important; flex-direction: column; text-align: left !important; align-items: stretch !important; padding: 0 !important; cursor: zoom-in; overflow: hidden;
        background: var(--spatchat-surface) !important; color: var(--spatchat-text) !important; border: 1px solid var(--spatchat-border); border-radius: 12px;
        box-shadow: var(--spatchat-shadow); transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease; font: inherit; }
    .spatchat-thumb:hover { transform: translateY(-2px); border-color: var(--spatchat-accent); box-shadow: 0 6px 18px rgba(16,36,26,.16); }
    .spatchat-thumb-preview { display: block; height: 100px; background: #fff; overflow: hidden; border-bottom: 1px solid var(--spatchat-border); }
    .spatchat-thumb-img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .spatchat-thumb-table { height: 100%; overflow: hidden; font-size: 9px; padding: 6px; color: #10241a; pointer-events: none; }
    .spatchat-thumb-meta { display: flex; flex-direction: column; gap: 2px; padding: 7px 10px 9px; text-align: left; }
    .spatchat-thumb-title { font-size: 12px; font-weight: 600; line-height: 1.3; }
    .spatchat-thumb-sub { font-size: 10.5px; color: var(--spatchat-text-secondary); line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .spatchat-thumb-badge { position: absolute; top: 8px; left: 8px; background: var(--spatchat-accent); color: #fff; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
    .spatchat-thumb-zoom { position: absolute; top: 8px; right: 8px; background: rgba(16,36,26,.78); color: #fff; font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 999px; opacity: 0; transition: opacity .12s; }
    .spatchat-thumb:hover .spatchat-thumb-zoom, .spatchat-thumb:focus-visible .spatchat-thumb-zoom { opacity: 1; }
    #spatchat-lightbox { position: fixed; inset: 0; z-index: 100000; display: none; align-items: center; justify-content: center; }
    #spatchat-lightbox.open { display: flex; }
    #spatchat-lightbox .sl-backdrop { position: absolute; inset: 0; background: rgba(8,14,11,.62); backdrop-filter: blur(2px); }
    #spatchat-lightbox .sl-panel { position: relative; display: flex; flex-direction: column; width: min(1100px, 92vw); height: min(86vh, 900px);
        background: var(--spatchat-surface, #fff); color: var(--spatchat-text, #10241a); border-radius: 16px; overflow: hidden;
        box-shadow: 0 24px 80px rgba(0,0,0,.45); border: 1px solid var(--spatchat-border, rgba(0,0,0,.1)); }
    #spatchat-lightbox .sl-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 16px;
        border-bottom: 1px solid var(--spatchat-border, rgba(0,0,0,.1)); background: var(--spatchat-surface-alt, #eef3ee); }
    #spatchat-lightbox .sl-title { font-size: 15px; font-weight: 700; }
    #spatchat-lightbox .sl-sub { font-size: 12px; color: var(--spatchat-text-secondary, #5c6b63); }
    #spatchat-lightbox .sl-nav { display: flex; align-items: center; gap: 6px; }
    #spatchat-lightbox .sl-count { font-size: 12px; color: var(--spatchat-text-secondary, #5c6b63); margin-right: 6px; }
    #spatchat-lightbox .sl-nav button { width: 34px; height: 34px; border-radius: 8px; border: 1px solid var(--spatchat-border, rgba(0,0,0,.12)); cursor: pointer;
        background: var(--spatchat-surface, #fff); color: var(--spatchat-text, #10241a); font-size: 17px; line-height: 1; }
    #spatchat-lightbox .sl-nav button:hover:not(:disabled) { background: var(--spatchat-accent-soft, #e3f2e8); border-color: var(--spatchat-accent, #1f7a4f); }
    #spatchat-lightbox .sl-nav button:disabled { opacity: .35; cursor: default; }
    #spatchat-lightbox .sl-body { flex: 1 1 auto; min-height: 0; overflow: auto; display: flex; align-items: flex-start; justify-content: center; padding: 16px; background: var(--spatchat-surface, #fff); }
    #spatchat-lightbox .sl-body img { max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; background: #fff; border-radius: 8px; margin: auto; }
    #spatchat-lightbox .sl-table { width: 100%; }
    body.spatchat-lightbox-open { overflow: hidden; }
    .spatchat-thumb-table table, #spatchat-lightbox .sl-table table { width: 100%; border-collapse: collapse; }
    .spatchat-thumb-table th, #spatchat-lightbox .sl-table th { background: var(--spatchat-surface-alt) !important; color: var(--spatchat-text) !important; font-weight: 600; text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--spatchat-border); }
    .spatchat-thumb-table td, #spatchat-lightbox .sl-table td { background: transparent !important; color: var(--spatchat-text) !important; padding: 4px 8px; border-bottom: 1px solid var(--spatchat-border); }
    .spatchat-thumb-table { background: var(--spatchat-surface); }
    /* ================= dark mode ================= */
    .spatchat-dark {
        --spatchat-connect-bg: #0e1512; --spatchat-surface: #16201b; --spatchat-surface-alt: #1e2a24;
        --spatchat-border: rgba(255,255,255,0.12); --spatchat-text: #e8efe9; --spatchat-text-secondary: #a6b4ab; --spatchat-text-subtle: #7f8f85;
        --spatchat-accent: #3fb27f; --spatchat-accent-hover: #5cc796; --spatchat-accent-soft: #1d3a2d;
        --spatchat-shadow: 0 1px 3px rgba(0,0,0,.5), 0 1px 2px rgba(0,0,0,.4);
    }
    body.spatchat-dark #spatchat-navbar a { color: var(--spatchat-text) !important; }
    body.spatchat-dark #spatchat-navbar a:hover { color: var(--spatchat-accent) !important; }
    body.spatchat-dark #spatchat-input-row { box-shadow: 0 2px 12px rgba(0,0,0,.5) !important; }
    body.spatchat-dark #spatchat-logo-img img, body.spatchat-dark #logo-img img { background: #fff; border-radius: 8px; padding: 2px 6px; }
    body.spatchat-dark .spatchat-map-empty-hint-card { background: var(--spatchat-surface) !important; color: var(--spatchat-text) !important; border-color: var(--spatchat-border) !important; }
    body.spatchat-dark #spatchat-datasheet, body.spatchat-dark #spatchat-datasheet * { color: var(--spatchat-text); }
    body.spatchat-dark #spatchat-datasheet th { background: var(--spatchat-surface-alt) !important; }
    body.spatchat-dark #spatchat-datasheet td { background: var(--spatchat-surface) !important; }
    body.spatchat-dark #spatchat-figure-layer .spatchat-figure-card-image { background: #fff; border-radius: 8px; }
    body.spatchat-dark .spatchat-figure-table th, body.spatchat-dark .spatchat-figure-table td { color: var(--spatchat-text) !important; }
    /* ================= Enterprise polish (keep last) ================= */
    :root {
        --spatchat-connect-bg: #f3f5f8;
        --spatchat-surface: #ffffff;
        --spatchat-surface-alt: #f1f4f8;
        --spatchat-border: rgba(15, 23, 42, 0.09);
        --spatchat-text: #0f172a;
        --spatchat-text-secondary: #475569;
        --spatchat-text-subtle: #94a3b8;
        --spatchat-accent: #0f766e;
        --spatchat-accent-hover: #115e59;
        --spatchat-accent-soft: #e6f4f1;
        --spatchat-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        --spatchat-card-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .gradio-container { padding-top: 10px !important; }
    /* blocks that only carry a <style> tag would otherwise add an empty row plus a gap each */
    .gradio-container .contain > .column > .block:has(style) { display: none !important; }
    #spatchat-header-row { border-radius: 12px !important; padding: 2px 16px !important; box-shadow: var(--spatchat-shadow) !important; }
    #spatchat-navbar nav { gap: 6px !important; }
    #spatchat-navbar a { color: var(--spatchat-text-secondary) !important; font-weight: 500 !important; padding: 6px 12px !important; border-radius: 8px; border-bottom: none !important; }
    #spatchat-navbar a:hover { background: var(--spatchat-surface-alt); color: var(--spatchat-text) !important; }
    #spatchat-navbar a[href$="/browse"] { color: var(--spatchat-accent) !important; background: var(--spatchat-accent-soft); }

    .spatchat-title-row { margin: 12px 2px 16px !important; align-items: center !important; }
    .spatchat-title { font-size: 22px !important; font-weight: 650 !important; letter-spacing: -0.015em !important; }
    .spatchat-subtitle { font-size: 13.5px !important; color: var(--spatchat-text-secondary) !important; margin-top: 3px !important; }
    #spatchat-theme-btn, #spatchat-share-btn { height: 34px; padding: 0 14px !important; font-size: 13px !important; font-weight: 600 !important; border-radius: 8px !important; }
    #spatchat-theme-btn { background: var(--spatchat-surface) !important; border: 1px solid var(--spatchat-border) !important; color: var(--spatchat-text) !important; box-shadow: var(--spatchat-shadow); }
    #spatchat-theme-btn:hover { background: var(--spatchat-surface-alt) !important; }
    #spatchat-share-btn { background: var(--spatchat-accent) !important; border: 1px solid var(--spatchat-accent) !important; color: #fff !important; }
    #spatchat-share-btn:hover { background: var(--spatchat-accent-hover) !important; border-color: var(--spatchat-accent-hover) !important; color: #fff !important; }

    #spatchat-workarea { background: transparent !important; height: calc(100vh - 150px) !important; min-height: 560px; padding-bottom: 12px !important; }
    #spatchat-workarea > .gradio-row { gap: 8px !important; }
    #spatchat-sidebar, #spatchat-mapcol {
        background: var(--spatchat-surface) !important; border: 1px solid var(--spatchat-border) !important;
        border-radius: 16px !important; box-shadow: var(--spatchat-card-shadow) !important;
    }
    #spatchat-sidebar { padding: 18px 18px 8px !important; }
    #spatchat-mapcol { padding: 12px 16px 16px !important; overflow: hidden !important; }
    #spatchat-splitter { background: transparent !important; }

    #spatchat-chatbot .message, #spatchat-chatbot .message * { font-size: 14.5px; line-height: 1.6; }
    #spatchat-chatbot .user, #spatchat-chatbot .user.message, #spatchat-chatbot .message.user {
        background: var(--spatchat-accent-soft) !important; border-radius: 14px 14px 4px 14px !important; padding: 9px 14px !important;
    }
    #spatchat-chatbot .message.panel-full-width, #spatchat-chatbot .bot .panel-full-width { background: transparent !important; }
    #spatchat-tabs > .tab-wrapper::after { display: none !important; }
    #spatchat-chips { gap: 6px !important; padding: 10px 0 12px !important; }
    .spatchat-chip, #spatchat-sample-btn.spatchat-chip {
        border-radius: 8px !important; padding: 5px 11px !important; font-size: 12.5px !important; font-weight: 500 !important;
        background: var(--spatchat-surface-alt) !important; border: 1px solid transparent !important; color: var(--spatchat-text-secondary) !important;
    }
    .spatchat-chip:hover { background: var(--spatchat-accent-soft) !important; border-color: transparent !important; color: var(--spatchat-accent) !important; }
    #spatchat-input-row { border-radius: 12px !important; box-shadow: var(--spatchat-shadow) !important; min-height: 48px; margin-bottom: 12px !important; }
    #spatchat-input-row:focus-within { border-color: var(--spatchat-accent) !important; box-shadow: 0 0 0 3px var(--spatchat-accent-soft) !important; }
    #spatchat-send-btn { border-radius: 10px !important; flex: 0 0 36px !important; width: 36px !important; min-width: 36px !important; height: 36px !important; }
    #spatchat-file-input { border-radius: 10px !important; }

    #spatchat-tabs > .tab-wrapper, #spatchat-tabs > .tab-container { border-bottom: 1px solid var(--spatchat-border) !important; }
    #spatchat-tabs button[role="tab"] { font-size: 13.5px !important; font-weight: 600 !important; padding: 8px 16px !important; border: none !important; border-bottom: 2px solid transparent !important; }
    #spatchat-tabs button[role="tab"].selected { color: var(--spatchat-accent) !important; border-bottom-color: var(--spatchat-accent) !important; }
    #spatchat-tabs .tabitem { border: none !important; box-shadow: none !important; margin-top: 12px !important; border-radius: 12px !important; }
    #spatchat-tabs #spatchat-map { border-radius: 12px; overflow: hidden; border: 1px solid var(--spatchat-border); }
    #spatchat-tabs div.tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane) { background: transparent !important; }
    body.spatchat-embedded #spatchat-header-row { display: none !important; }
    body.spatchat-embedded .gradio-container .main { padding-top: 0 !important; }
    body.spatchat-embedded .gradio-container { padding-top: 0 !important; }
    body.spatchat-embedded .spatchat-title-row { margin-top: 6px !important; }
    body.spatchat-embedded #spatchat-workarea { height: calc(100vh - 135px) !important; min-height: 520px; }
    html.spatchat-embedded:not(.spatchat-dark), body.spatchat-embedded:not(.spatchat-dark),
    body.spatchat-embedded:not(.spatchat-dark) gradio-app,
    body.spatchat-embedded:not(.spatchat-dark) .gradio-container,
    body.spatchat-embedded:not(.spatchat-dark) .gradio-container .main,
    body.spatchat-embedded:not(.spatchat-dark) #spatchat-workarea, body.spatchat-embedded:not(.spatchat-dark) #spatchat-splitter { background: transparent !important; }
    body.spatchat-dark #spatchat-sidebar, body.spatchat-dark #spatchat-mapcol, body.spatchat-dark #spatchat-workarea, body.spatchat-dark #spatchat-splitter { background: var(--spatchat-connect-bg) !important; }
    

/* ---- single slim top bar: logo, room name, links, dark mode, share ---- */
#spatchat-topbar { display: flex; align-items: center; gap: 12px; background: var(--spatchat-surface); border: 1px solid var(--spatchat-border);
  border-radius: 12px; padding: 6px 14px; margin: 4px 0 10px; box-shadow: var(--spatchat-shadow); }
#spatchat-topbar .sp-brand { display: flex; align-items: center; }
#spatchat-topbar .sp-brand img { height: 28px; width: auto; display: block; }
body.spatchat-dark #spatchat-topbar .sp-brand img { background: #fff; border-radius: 6px; padding: 2px 6px; }
#spatchat-topbar .sp-room { margin: 0 !important; font-size: 17px !important; font-weight: 650 !important; letter-spacing: -0.01em;
  color: var(--spatchat-text) !important; line-height: 1.2; }
#spatchat-topbar .sp-spacer { flex: 1; }
#spatchat-topbar .sp-link { font-size: 13px; font-weight: 600; color: var(--spatchat-text-secondary); text-decoration: none; padding: 6px 10px; border-radius: 8px; }
#spatchat-topbar .sp-link:hover { background: var(--spatchat-surface-alt); color: var(--spatchat-text); }
#spatchat-topbar .sp-gated { display: none; }
body.spatchat-gated #spatchat-topbar .sp-gated { display: inline-block; }
#spatchat-topbar .spatchat-title-actions { display: flex; gap: 8px; align-items: center; }
body.spatchat-embedded #spatchat-topbar { background: transparent; border: none; box-shadow: none; padding-left: 2px; }
body.spatchat-embedded #spatchat-topbar .sp-brand, body.spatchat-embedded #spatchat-topbar .sp-link { display: none !important; }
body #spatchat-workarea { height: calc(100vh - 96px) !important; min-height: 520px; }
body.spatchat-embedded #spatchat-workarea { height: calc(100vh - 84px) !important; }
</style>
"""


def _accepts(component, kw: str) -> bool:
    try:
        return kw in inspect.signature(component).parameters
    except (TypeError, ValueError):
        return False


def _html(gr, value, **kw):
    """gr.HTML with layout kwargs that older Gradio versions do not accept dropped."""
    for drop in ((), ("scale", "min_width")):
        try:
            return gr.HTML(value=value, **{k: v for k, v in kw.items() if k not in drop})
        except TypeError:
            continue
    return gr.HTML(value=value)


def build_header(gr, title: str, subtitle: str, room: str) -> None:
    """Emit the stylesheet and one slim top bar. Call first inside `with gr.Blocks(...)`.

    `subtitle` is kept for callers but no longer shown, to save vertical space."""
    gr.HTML(CSS)
    _html(gr, f"""
    <div id="spatchat-topbar">
      <a class="sp-brand" href="https://spatchat.org/" target="_blank" rel="noopener" title="SpatChat home">{logo_html()}</a>
      <h1 class="sp-room">{title}</h1>
      <span class="sp-spacer"></span>
      <a class="sp-link" href="https://spatchat.org/browse" target="_blank" rel="noopener">All rooms</a>
      <a class="sp-link sp-gated" href="/account">Account</a>
      <div class="spatchat-title-actions">
        <button id="spatchat-theme-btn" type="button" onclick="window.spatchatToggleTheme && window.spatchatToggleTheme()">
          <span class="spatchat-theme-light">Dark mode</span><span class="spatchat-theme-dark">Light mode</span>
        </button>
        <button id="spatchat-share-btn" type="button" onclick="
          navigator.clipboard.writeText('https://spatchat.org/browse/?room={room}');
          this.textContent = 'Link copied';
          setTimeout(() => {{ this.textContent = 'Share'; }}, 1500);
        ">Share</button>
      </div>
    </div>
    """, elem_id="spatchat-topbar-wrap")


def build_footer(gr, cite: str) -> None:
    gr.HTML(f"""
    <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(16,36,26,0.1); font-size: 11px; color: #5c6b63; text-align: center;">
      &copy; 2025 Ho Yi Wan &amp; Logan Hysen &middot; Cite: {cite}
    </div>
    """)


def chatbot_kwargs(gr, welcome: str, **extra) -> dict:
    """Keyword arguments for the Move-style chat window (assistant text on the page, soft user bubbles)."""
    kwargs = {
        "label": "Spatchat",
        "show_label": False,
        "layout": "bubble",
        "value": [{"role": "assistant", "content": welcome}],
        "elem_id": "spatchat-chatbot",
    }
    if _accepts(gr.Chatbot, "type"):
        kwargs["type"] = "messages"
    if _accepts(gr.Chatbot, "buttons"):
        kwargs["buttons"] = []
    if _accepts(gr.Chatbot, "feedback_options"):
        kwargs["feedback_options"] = None
    kwargs.update(extra)
    return kwargs


def chip_js(text: str) -> str:
    return (
        "() => { const ta = document.querySelector('#spatchat-user-input textarea, #spatchat-user-input input');"
        " if (!ta) return; ta.focus(); ta.value = " + json.dumps(text) + ";"
        " ta.dispatchEvent(new Event('input', { bubbles: true }));"
        " setTimeout(() => { const b = document.querySelector('#spatchat-send-btn'); if (b) b.click(); }, 80); }"
    )


def chip_row(gr, labels: list[str]):
    """Suggestion chips that type their label into the composer and send it."""
    buttons = []
    with gr.Row(elem_id="spatchat-chips"):
        for label in labels:
            btn = gr.Button(label, size="sm", elem_classes=["spatchat-chip"], scale=0, min_width=0)
            btn.click(fn=None, js=chip_js(label))
            buttons.append(btn)
    return buttons


def composer(gr, placeholder: str, file_types: list[str], *, file_count: str = "single"):
    """The pill-shaped message box: '+' upload button, text box, send arrow. Returns (upload, textbox, send)."""
    with gr.Row(elem_id="spatchat-input-row"):
        upload = gr.UploadButton("+", file_types=file_types, file_count=file_count, elem_id="spatchat-file-input", scale=0, min_width=44)
        textbox = gr.Textbox(label="", show_label=False, placeholder=placeholder, lines=1, elem_id="spatchat-user-input", scale=1)
        send = gr.Button("↑", elem_id="spatchat-send-btn", scale=0, min_width=40)
    return upload, textbox, send
