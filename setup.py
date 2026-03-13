from setuptools import setup, find_packages

setup(
    name="clawmory_rlm",
    version="0.1.0",
    description="Scalable Conversational Memory via Recursive Sub-Agent Delegation",
    author="Sayed Raheel Hussain",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv",
        "tenacity"
    ],
)
