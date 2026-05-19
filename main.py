from dotenv import load_dotenv
load_dotenv()

from agent import DataAnalystAgent

if __name__ == "__main__":
    DataAnalystAgent().run()
