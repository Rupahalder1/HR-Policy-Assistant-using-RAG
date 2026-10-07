"""Command-line demo of the HR Policy Assistant.

Run with:  python main.py
"""

from pipeline import ask , build_HR_Assistant

def main():
    print("Building the HR policy assistant...")
    agent = build_HR_Assistant()
    print("Assistant ready!\n")

    demo_questions = [
        "How many paid annual leave days do I get?",
        "What is the notice period during probation?",
        "Can I work from home every day?",
    ]

    for question in demo_questions:
        print("=" * 60)
        print("QUESTION:", question)
        print("-" * 60)
        answer = ask(agent, question)
        print("ANSWER:", answer)
        print("=" * 60)
        print()



if __name__ == "__main__":
    main()