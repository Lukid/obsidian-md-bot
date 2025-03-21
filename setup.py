from setuptools import setup, find_packages

setup(
    name="obsidian-md-bot",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "obsidian-md-bot=src.main:main",
        ],
    },
    install_requires=[
        "gradio>=5.22.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "markdown>=3.5.0",
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "python-dotenv>=1.0.0",
        "openai>=1.12.0",
        "anthropic>=0.20.0",
        "google-generativeai>=0.3.2",
        "sentence-transformers>=2.5.0"
    ],
    author="Luca Baroncini",
    description="Un'app Gradio per il webscraping e la creazione automatica di note Obsidian",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lucabaroncini",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9,<4.0",
)
