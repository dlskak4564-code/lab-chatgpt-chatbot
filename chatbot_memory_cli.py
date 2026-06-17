from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"
SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하세요."

history = []
usage_log = []


def reset():
    history.clear()
    usage_log.clear()
    print("(대화 초기화)")


def chat(message: str, temperature: float = 0.3) -> str:
    history.append({"role": "user", "content": message})
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM}, *history],
    )
    answer = resp.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    usage_log.append(resp.usage.total_tokens)
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

        response = chat(user_input)
        print("Model>", response)


if __name__ == "__main__":
    main()
