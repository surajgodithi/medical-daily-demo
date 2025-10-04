import json
from pathlib import Path

from medical_daily_agent_demo_patched import MedicalDailyAgent

CACHE_DIR = Path('medical_daily_cache')


def _load_latest_digest(agent: MedicalDailyAgent) -> Path:
    if not CACHE_DIR.exists():
        raise FileNotFoundError("Cache directory not found. Run the daily digest generator first.")

    digest_files = sorted(CACHE_DIR.glob('digest_*.json'), reverse=True)
    if not digest_files:
        raise FileNotFoundError("No digest metadata found. Run the daily digest generator first.")

    latest = digest_files[0]
    data = json.loads(latest.read_text())

    agent.articles = data.get('articles', [])
    agent.podcast_script = data.get('script', '')
    agent.audio_file = data.get('audio_file')
    agent.audio_url = data.get('audio_url')
    agent.audio_page_url = data.get('audio_page_url')

    return latest


def main() -> None:
    agent = MedicalDailyAgent()
    try:
        metadata_path = _load_latest_digest(agent)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return

    print(f"[INFO] Loaded digest from {metadata_path.name}")
    print("Type a command (medical daily / text / links / detail N). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input('You: ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\n[INFO] Ending session.')
            break

        if not user_input:
            continue

        if user_input.lower() in {'exit', 'quit'}:
            print('[INFO] Ending session.')
            break

        reply = agent.handle_message(user_input)
        print(f"Agent: {reply}\n")


if __name__ == '__main__':
    main()
