import re
import streamlit as st
import pandas as pd
import psycopg2
from datetime import datetime
from io import BytesIO
import google.generativeai as genai
from psycopg2.extras import RealDictCursor
import warnings
import logging
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import json

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('tornado.iostream').setLevel(logging.ERROR)

# Configure page
st.set_page_config(
    page_title="Text-to-SQL with LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# State definition for LangGraph
class SQLGenerationState(TypedDict):
    question: str
    schema: str
    generated_sql: str
    is_valid: bool
    validation_errors: str
    execution_result: dict
    error: str
    attempt: int

# Initialize session state
def init_session_state():
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'db_connection' not in st.session_state:
        st.session_state.db_connection = None
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = None
    if 'sql_generator_graph' not in st.session_state:
        st.session_state.sql_generator_graph = build_sql_graph()

# Database connection class
class PostgresConnection:
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.connect()
    
    def connect(self):
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            return True
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return False
    
    def get_schema(self):
        """Get database schema"""
        try:
            cursor = self.connection.cursor()
            
            query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
            """
            cursor.execute(query)
            tables = cursor.fetchall()
            
            schema = "# Database Schema\n\n"
            
            for table in tables:
                table_name = table[0]
                schema += f"## Table: {table_name}\n"
                
                col_query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """
                cursor.execute(col_query, (table_name,))
                columns = cursor.fetchall()
                
                for col in columns:
                    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                    schema += f"- {col[0]}: {col[1]} ({nullable})\n"
                
                schema += "\n"
            
            cursor.close()
            return schema
        except Exception as e:
            st.error(f"Failed to retrieve schema: {str(e)}")
            return None
    
    def execute_query(self, sql):
        """Execute SQL query and return results"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql)
            
            if sql.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                return {
                    'success': True,
                    'data': results,
                    'columns': columns,
                    'row_count': len(results)
                }
            else:
                self.connection.commit()
                cursor.close()
                return {
                    'success': True,
                    'row_count': cursor.rowcount,
                    'message': f"Query executed successfully"
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'row_count': 0
            }
    
    def close(self):
        if self.connection:
            self.connection.close()

# LangGraph nodes
def generate_sql_node(state: SQLGenerationState) -> SQLGenerationState:
    """Node 1: Generate SQL using Gemini via LangChain"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=st.session_state.gemini_api_key,
            temperature=0
        )
        
        prompt = f"""You are an expert SQL generator for PostgreSQL. Convert the following natural language question into a valid SQL query.

Database Schema:
{state['schema']}

Rules:
1. Generate ONLY valid PostgreSQL syntax
2. Return ONLY the SQL query without any explanation or markdown
3. Use appropriate JOINs if needed
4. Include proper filtering and aggregation
5. Make the query efficient
6. Do NOT wrap in code blocks - return raw SQL

Question: {state['question']}

SQL Query:"""

        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        sql = response.content.strip()
        
        # Clean up if wrapped in code blocks
        if '```sql' in sql:
            sql = sql.split('```sql')[1].split('```')[0].strip()
        elif '```' in sql:
            sql = sql.split('```')[1].split('```')[0].strip()
        
        state['generated_sql'] = sql
        state['error'] = ""
        
    except Exception as e:
        state['error'] = f"SQL Generation failed: {str(e)}"
    
    return state

def validate_sql_node(state: SQLGenerationState) -> SQLGenerationState:
    """Node 2: Validate generated SQL"""
    try:
        sql = state['generated_sql'].strip()
        
        # Basic SQL validation
        validation_checks = [
            (sql.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE')), "Query must start with SELECT, INSERT, UPDATE, or DELETE"),
            (sql.endswith(';') or len(sql) > 0, "Query should be valid SQL"),
            (sql.count('SELECT') == 1, "Query should have only one SELECT statement"),
        ]
        
        errors = []
        for check, message in validation_checks:
            if not check:
                errors.append(message)
        
        if errors:
            state['is_valid'] = False
            state['validation_errors'] = "; ".join(errors)
        else:
            state['is_valid'] = True
            state['validation_errors'] = ""
        
    except Exception as e:
        state['is_valid'] = False
        state['validation_errors'] = str(e)
    
    return state

def fix_sql_node(state: SQLGenerationState) -> SQLGenerationState:
    """Node 3: If validation fails, attempt to fix SQL"""
    if state['is_valid']:
        return state
    
    try:
        if state['attempt'] >= 2:
            state['error'] = "Max retry attempts reached"
            return state
        
        state['attempt'] += 1
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=st.session_state.gemini_api_key,
            temperature=0
        )
        
        fix_prompt = f"""Fix this invalid SQL query. The error was: {state['validation_errors']}

Original question: {state['question']}
Database Schema:
{state['schema']}

Invalid SQL:
{state['generated_sql']}

Provide ONLY the corrected SQL query without any explanation or markdown:"""

        message = HumanMessage(content=fix_prompt)
        response = llm.invoke([message])
        
        sql = response.content.strip()
        
        if '```sql' in sql:
            sql = sql.split('```sql')[1].split('```')[0].strip()
        elif '```' in sql:
            sql = sql.split('```')[1].split('```')[0].strip()
        
        state['generated_sql'] = sql
        
        # Re-validate
        state = validate_sql_node(state)
        
    except Exception as e:
        state['error'] = f"SQL fixing failed: {str(e)}"
    
    return state

def execute_sql_node(state: SQLGenerationState) -> SQLGenerationState:
    """Node 4: Execute the validated SQL"""
    if not state['is_valid']:
        state['execution_result'] = {
            'success': False,
            'error': state['validation_errors']
        }
        return state
    
    try:
        result = st.session_state.db_connection.execute_query(state['generated_sql'])
        state['execution_result'] = result
        
    except Exception as e:
        state['execution_result'] = {
            'success': False,
            'error': str(e)
        }
    
    return state

def build_sql_graph():
    """Build LangGraph workflow for SQL generation"""
    workflow = StateGraph(SQLGenerationState)
    
    # Add nodes
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("fix_sql", fix_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    
    # Add edges
    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_edge("validate_sql", "fix_sql")
    workflow.add_edge("fix_sql", "execute_sql")
    workflow.add_edge("execute_sql", END)
    
    return workflow.compile()

# Sidebar
def render_sidebar():
    st.sidebar.header("🔌 Database Configuration")
    
    with st.sidebar.expander("➕ Database Connection", expanded=False):
        with st.form("db_connection_form"):
            host = st.text_input("Host", value="localhost")
            port = st.text_input("Port", value="5432")
            database = st.text_input("Database", placeholder="postgres")
            user = st.text_input("Username", placeholder="postgres")
            password = st.text_input("Password", type="password")
            
            submitted = st.form_submit_button("🔗 Connect", use_container_width=True)
            
            if submitted:
                if all([host, port, database, user, password]):
                    try:
                        conn = PostgresConnection(
                            host=host,
                            port=int(port),
                            database=database,
                            user=user,
                            password=password
                        )
                        st.session_state.db_connection = conn
                        st.success(f"✅ Connected to {database}")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
                else:
                    st.error("Please fill all fields")
    
    st.sidebar.divider()
    st.sidebar.subheader("🤖 Gemini Configuration")
    
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if api_key:
        st.session_state.gemini_api_key = api_key
        st.sidebar.success("✅ API Key configured")
    
    st.sidebar.divider()
    st.sidebar.subheader("📊 Status")
    
    if st.session_state.db_connection:
        st.sidebar.success(f"✅ Database Connected")
        if st.sidebar.button("View Schema", use_container_width=True):
            st.session_state.show_schema = True
    else:
        st.sidebar.warning("⚠️ No database connection")

# Main interface
def render_main_interface():
    st.title("🤖 Text-to-SQL with LangGraph")
    
    if not st.session_state.db_connection:
        st.warning("⚠️ Please configure database connection in the sidebar")
        return
    
    if not st.session_state.gemini_api_key:
        st.warning("⚠️ Please add Gemini API Key in the sidebar")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"🗄️ Database: **{st.session_state.db_connection.database}**")
    with col2:
        if st.button("📋 View Schema", use_container_width=True):
            st.session_state.show_schema = True
    
    if st.session_state.get('show_schema', False):
        with st.expander("📋 Database Schema", expanded=True):
            schema = st.session_state.db_connection.get_schema()
            if schema:
                st.markdown(schema)
        st.session_state.show_schema = False
    
    st.subheader("💬 Ask Your Question")
    
    user_question = st.text_area(
        "Enter your question in natural language:",
        height=100,
        placeholder="e.g., Show me all customers who made purchases in the last 30 days"
    )
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        generate_button = st.button("🚀 Generate SQL", type="primary", use_container_width=True)
    with col2:
        execute_button = st.button("▶️ Generate & Execute", use_container_width=True)
    with col3:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if (generate_button or execute_button) and user_question:
        process_query_with_langgraph(user_question, execute=execute_button)
    
    if st.session_state.query_history:
        st.divider()
        st.subheader("📜 Query History")
        
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            with st.expander(
                f"{item['question'][:80]}..." if len(item['question']) > 80 
                else item['question'],
                expanded=(i == 0)
            ):
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Timestamp:** {item['timestamp']}")
                st.code(item['sql'], language='sql')
                
                if 'result' in item:
                    if item['result']['success']:
                        st.success(f"✅ Returned {item['result']['row_count']} rows")
                    else:
                        st.error(f"❌ {item['result']['error']}")

def process_query_with_langgraph(question: str, execute: bool = False):
    """Process query using LangGraph workflow"""
    try:
        with st.spinner("📊 Analyzing database schema..."):
            schema = st.session_state.db_connection.get_schema()
        
        if not schema:
            st.error("Failed to retrieve schema")
            return
        
        # Initialize state
        initial_state: SQLGenerationState = {
            'question': question,
            'schema': schema,
            'generated_sql': '',
            'is_valid': False,
            'validation_errors': '',
            'execution_result': {},
            'error': '',
            'attempt': 0
        }
        
        # Run the graph
        with st.spinner("🧠 Processing with LangGraph..."):
            result = st.session_state.sql_generator_graph.invoke(initial_state)
        
        # Display results
        st.subheader("✨ Generated SQL Query")
        st.code(result['generated_sql'], language='sql')
        
        if result['validation_errors']:
            st.warning(f"⚠️ Validation Notes: {result['validation_errors']}")
        
        history_item = {
            'question': question,
            'sql': result['generated_sql'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if result['is_valid']:
            st.success("✅ SQL is valid and ready to execute")
            
            if execute or st.button("Execute Query"):
                display_results(result['execution_result'], history_item)
        else:
            st.error(f"❌ SQL validation failed: {result['validation_errors']}")
            history_item['result'] = result['execution_result']
            st.session_state.query_history.append(history_item)
    
    except Exception as e:
        st.error(f"❌ Error processing query: {str(e)}")

def display_results(result: dict, history_item: dict):
    """Display query results"""
    history_item['result'] = result
    st.session_state.query_history.append(history_item)
    
    if result['success']:
        st.success("✅ Query executed successfully!")
        
        if result['row_count'] > 0 and 'data' in result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", result['row_count'])
            with col2:
                st.metric("Columns", len(result['columns']))
            with col3:
                st.metric("Status", "Success")
            
            df = pd.DataFrame(result['data'])
            
            st.subheader("📊 Query Results")
            st.dataframe(df, use_container_width=True, height=min(400, 50 + len(df) * 35))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                st.download_button(
                    label="📥 Download as Excel",
                    data=output.getvalue(),
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col3:
                json_data = json.dumps(result['data'], indent=2, default=str)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        elif result['row_count'] == 0:
            st.info("Query executed but returned no results.")
    else:
        st.error(f"❌ Query execution failed: {result['error']}")

def main():
    init_session_state()
    render_sidebar()
    render_main_interface()

if __name__ == "__main__":
    main()