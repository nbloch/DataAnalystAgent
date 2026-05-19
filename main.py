import argparse

from dotenv import load_dotenv
load_dotenv()

from agent import DataAnalystAgent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None, help="Session ID for conversation memory")
    args = parser.parse_args()
    DataAnalystAgent().run(session_id=args.session)
