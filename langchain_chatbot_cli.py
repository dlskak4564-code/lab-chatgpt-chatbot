from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
MODEL = "gpt-4o-mini"
SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하세요."

history = []
usage_log = []


def reset():
    history.clear()
    usage_log.clear()
    print("(대화 초기화)")


def _to_langchain_messages():
    msgs = [SystemMessage(content=SYSTEM)]
    for item in history:
        cls = HumanMessage if item["role"] == "user" else AIMessage
        msgs.append(cls(content=item["content"]))
    return msgs


def chat(message: str, temperature: float = 0.3) -> str:
    history.append({"role": "user", "content": message})
    llm = ChatOpenAI(model=MODEL, temperature=temperature)
    resp = llm.invoke(_to_langchain_messages())
    answer = resp.content or ""
    history.append({"role": "assistant", "content": answer})

    tokens = None
    usage_metadata = getattr(resp, "usage_metadata", None)
    if usage_metadata:
        tokens = usage_metadata.get("total_tokens")
    else:
        response_metadata = getattr(resp, "response_metadata", None) or {}
        tokens = response_metadata.get("token_usage", {}).get("total_tokens")
    usage_log.append(tokens)
    return answer


def main():
    print("멀티턴 CLI 챗봇에 오신 것을 환영합니다.")
    print("/reset 입력 시 대화 초기화, /exit 또는 빈 줄 입력 시 종료합니다.")

    while True:
        try:
            user_input = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n대화를 종료합니다.")
            break

        if not user_input:
            print("대화를 종료합니다.")
            break

        normalized = user_input.lower()
        if normalized == "/reset":
            reset()
            continue
        if normalized in {"/exit", "/quit", "bye", "종료"}:
            print("대화를 종료합니다.")
            break

        try:
            response = chat(user_input)
        except Exception:
            print("Model> 죄송합니다. 일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요.")
            continue

        print("Model>", response)


if __name__ == "__main__":
    main()
