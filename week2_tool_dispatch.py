import json
import os
import requests

API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "minimaxai/minimax-m3.1-flash"


def http_get(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15)
        snippet = resp.text[:500].replace("\n", " ")
        result = {
            "url": url,
            "status_code": resp.status_code,
            "body_snippet": snippet,
        }
    except Exception as exc:
        result = {
            "url": url,
            "error": str(exc),
        }
    return json.dumps(result, ensure_ascii=False)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "對指定網址發送 HTTP GET 請求並取得狀態碼與網頁內容摘要",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要請求的完整網址，例如 https://example.com",
                    }
                },
                "required": ["url"],
            },
        },
    }
]


def dispatch(name: str, arguments_json: str) -> str:
    """依模型建議的工具名稱與參數分派執行。"""
    args = json.loads(arguments_json) if arguments_json else {}
    if name == "http_get":
        return http_get(args["url"])
    return json.dumps({"error": f"未知的工具: {name}"}, ensure_ascii=False)


def call_chat_completions(messages: list, tools: list | None = None) -> dict:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("找不到環境變數NVIDIA_API_KEY，請先設定。")

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 1,
        "top_p": 0.95,
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    print("HTTP status:", resp.status_code)
    data = resp.json()
    if not resp.ok:
        print("API error:", json.dumps(data, ensure_ascii=False, indent=2))
    return data


def main():
    messages = [
        {
            "role": "user",
            "content": "請使用http_get工具查看 https://example.com 的回應狀態與內容摘要，再用繁體中文簡短說明。",
        }
    ]

    print("=== 第 1 次呼叫：帶入 tools，取得模型的 tool_calls ===")
    first_data = call_chat_completions(messages, tools=TOOLS)
    first_msg = first_data["choices"][0]["message"]
    print("Assistant 第 1 輪訊息：", json.dumps(first_msg, ensure_ascii=False, indent=2))

    tool_calls = first_msg.get("tool_calls") or []
    if not tool_calls:
        print("模型這次沒有呼叫工具，直接回覆：", first_msg.get("content"))
        return

    messages.append(first_msg)

    
    for tc in tool_calls:
        tc_id = tc["id"]
        fn_name = tc["function"]["name"]
        fn_args = tc["function"]["arguments"]
        print(f"\n[程式執行工具] name={fn_name}, arguments={fn_args}")

        tool_output = dispatch(fn_name, fn_args)
        print(f"[工具實際回傳] {tool_output}")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": tool_output,
            }
        )

    print("\n=== 第 2 次呼叫：帶入 tool 結果，取得最終回覆 ===")
    second_data = call_chat_completions(messages, tools=TOOLS)
    second_msg = second_data["choices"][0]["message"]
    print("最終回覆：", second_msg.get("content"))


if __name__ == "__main__":
    main()
