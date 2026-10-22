## Environment Setup

This project uses a **virtual environment** to keep dependencies isolated and reproducible.

1.  Create and activate the environment/ install new dependencies by running `setup_env.py`. Run `setup_env.py` whenever you pull new changes or start work to stay in sync with the latest dependencies.

2.  Add new libraries as needed:

    ``` python
    pip install <package-name>
    pip freeze > requirements.txt
    ```
