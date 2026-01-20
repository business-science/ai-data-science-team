# Created a contributing.md file 

# Contributing to AI Data Science Team

Thank you for your interest in contributing to the AI Data Science Team project! We welcome contributions from the community to help make this project better.

## Getting Started

To get started with contributing, follow these steps:

1.  **Fork the repository**: Click the "Fork" button on the top right corner of the repository page to create a copy of the repository in your GitHub account.
2.  **Clone the repository**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/YOUR_USERNAME/ai-data-science-team.git
    cd ai-data-science-team
    ```
3.  **Set up the environment**:
    It is recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

    Install the package in editable mode with dependencies:
    ```bash
    pip install -e .
    ```

## Running the App

To run the AI Pipeline Studio app locally:

```bash
streamlit run apps/ai-pipeline-studio-app/app.py
```

## Reporting Bugs

If you find a bug, please open an issue in the issue tracker. Include as much detail as possible, such as:
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Your environment details (OS, Python version, etc.)

## Submitting Pull Requests

1.  **Create a branch**: Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b feature/your-feature-name
    ```
2.  **Make changes**: Make your changes to the code.
3.  **Test your changes**: Ensure that your changes work as expected and do not break existing functionality.
4.  **Commit your changes**: Commit your changes with a descriptive commit message.
    ```bash
    git commit -m "Description of your changes"
    ```
5.  **Push to GitHub**: Push your changes to your forked repository.
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **Create a Pull Request**: Go to the original repository and click "Compare & pull request" to submit your changes for review.

We appreciate your contributions!
