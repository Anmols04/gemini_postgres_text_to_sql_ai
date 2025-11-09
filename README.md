# Text-to-SQL with LangGraph and Gemini 🤖📊

This Streamlit application leverages **LangGraph** and Google's **Gemini 2.0 Flash** model to convert natural language questions into executable SQL queries for PostgreSQL databases. It features a self-correcting workflow that validates and attempts to fix generated SQL before execution.

## ✨ Features

* **LangGraph Workflow**: Orchestrated flow for Generation → Validation → Fixing → Execution.
* **Self-Correcting SQL**: Automatically attempts to fix invalid SQL syntax using Gemini.
* **Schema Awareness**: Fetches and utilizes the actual database schema for accurate query generation.
* **Interactive UI**: Built with Streamlit for easy database connection and querying.
* **Multiple Export Formats**: Download query results as CSV, Excel, or JSON.
* **Query History**: Keeps track of your recent questions and their results.

## 🛠️ Prerequisites

Before running the application, ensure you have the following:

* **Python 3.9+** installed on your system.
* A running **PostgreSQL** database that you can connect to.
* A **Google Gemini API Key** (specifically with access to `gemini-2.0-flash`).

## 🔑 How to Get a Google Gemini API Key

To use this application, you need an API key from Google AI Studio.

1.  Go to [Google AI Studio](https://aistudio.google.com/).
2.  Sign in with your Google Account.
3.  On the left sidebar, click on **"Get API key"**.
4.  Click **"Create API key"**.
5.  You can choose to create a key in a new project or an existing Google Cloud project.
6.  Copy the generated API key. You will need to enter this key in the application sidebar.

## 📦 Installation

1.  **Clone this repository** (or download the `gemini2.py` file):
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run gemini2.py
    ```

2.  **Configure the Application** (in the left sidebar):
    * **Database Connection**: Enter your PostgreSQL host, port (default 5432), database name, username, and password. Click "🔗 Connect".
    * **Gemini Configuration**: Paste your Google Gemini API Key in the designated field.

3.  **Ask Questions**:
    * Once connected, enter a natural language question in the main text area (e.g., "Show me top 5 customers by total sales revenue").
    * Click **"🚀 Generate SQL"** to see the query first, or **"▶️ Generate & Execute"** to run it immediately.

## 🧩 Architecture

The application uses `LangGraph` to define a stateful workflow:

1.  **Generate Node**: Converts the user question and DB schema into an initial SQL query using Gemini.
2.  **Validate Node**: Performs basic syntax checks (ensures it starts with SELECT/INSERT/etc.).
3.  **Fix Node**: If validation fails, it asks Gemini to correct the SQL based on the error message (up to 2 retries).
4.  **Execute Node**: Runs the valid SQL against the PostgreSQL database and returns results.
