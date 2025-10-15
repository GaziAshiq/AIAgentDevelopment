import os
import re
import asyncio
from typing import Optional, Dict, Any
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine, text, pool
from sqlalchemy.exc import SQLAlchemyError
from agents import Agent, Runner, function_tool
from loguru import logger

load_dotenv(override=True)


def _error_response(error: str) -> Dict[str, Any]:
    """Helper to create error response dict"""
    return {"success": False, "error": error, "data": [], "columns": [], "row_count": 0}


class DBManager:
    """PostgreSQL connection manager with pooling and read-only query execution"""

    ALLOWED = [r"^\s*SELECT\s+", r"^\s*EXPLAIN\s+", r"^\s*WITH\s+.*\s+SELECT\s+"]
    FORBIDDEN = [r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE)\b",
                 r";.*SELECT", r"--", r"/\*.*\*/"]

    def __init__(self, connection_url: Optional[str] = None, pool_size: int = 5,
                 max_overflow: int = 10, pool_timeout: int = 30, pool_recycle: int = 3600):
        self.url = connection_url or os.getenv("DATABASE_URL")
        if not self.url:
            raise ValueError("DATABASE_URL not set. Add it to .env file.")

        self._engine = create_engine(
            self.url, poolclass=pool.QueuePool, pool_size=pool_size,
            max_overflow=max_overflow, pool_timeout=pool_timeout,
            pool_recycle=pool_recycle, echo=False
        )

        # Test connection on startup
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connected with pooling")
        except SQLAlchemyError as e:
            logger.error(f"Connection test failed: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self._engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def _validate_query(self, query: str) -> tuple[bool, str]:
        """Validate query is read-only and safe"""
        q = query.upper().strip()
        if not any(re.match(p, q, re.IGNORECASE) for p in self.ALLOWED):
            return False, "Only SELECT/EXPLAIN queries allowed"
        if any(re.search(p, q, re.IGNORECASE) for p in self.FORBIDDEN):
            return False, "Forbidden operation detected"
        return True, ""

    def execute_query(self, query: str, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """Execute validated read-only SQL query"""
        if not (is_valid := self._validate_query(query))[0]:
            logger.warning(f"Invalid query: {is_valid[1]}")
            return _error_response(is_valid[1])

        try:
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                rows = result.fetchmany(max_rows) if max_rows else result.fetchall()
                columns = list(result.keys()) if result.keys() else []
                data = [dict(zip(columns, row)) for row in rows]
                logger.info(f"Query executed: {len(data)} rows")
                return {"success": True, "data": data, "columns": columns,
                       "row_count": len(data), "error": None}
        except SQLAlchemyError as e:
            logger.error(f"Query failed: {e}")
            return _error_response(str(e))

    def get_schema(self, table: str) -> Dict[str, Any]:
        """Get table schema - validates table name to prevent SQL injection"""
        # Sanitize table name (alphanumeric and underscore only)
        if not re.match(r'^[a-zA-Z0-9_]+$', table):
            return _error_response("Invalid table name")
        return self.execute_query(f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = '{table}' ORDER BY ordinal_position
        """)

    def list_tables(self) -> Dict[str, Any]:
        """List all tables in public schema"""
        return self.execute_query("""
            SELECT table_name, table_type FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
        """)

    def close(self):
        """Dispose engine and close connections"""
        if self._engine:
            self._engine.dispose()
            logger.info("Database closed")


# Initialize global DB manager
try:
    db = DBManager()
except ValueError as e:
    logger.error(f"Database setup error: {e}")
    db = None


def _check_db(operation: str) -> Optional[str]:
    """Helper to check if database is configured"""
    return None if db else f"Error: Database not configured for {operation}"


# Agent tool functions
@function_tool
def execute_sql_query(query: str, max_rows: Optional[int] = None) -> str:
    """Execute read-only SQL query. Supports SELECT, JOINs, aggregations, CTEs."""
    if err := _check_db("query execution"):
        return err
    result = db.execute_query(query, max_rows)
    return (f"Success! {result['row_count']} rows.\nColumns: {result['columns']}\nData: {result['data']}"
            if result["success"] else f"Query failed: {result['error']}")


@function_tool
def get_table_schema(table_name: str) -> str:
    """Get table schema (columns, types, nullability)"""
    if err := _check_db("schema retrieval"):
        return err
    result = db.get_schema(table_name)
    return f"Schema for '{table_name}':\n{result['data']}" if result["success"] else f"Failed: {result['error']}"


@function_tool
def list_all_tables() -> str:
    """List all database tables"""
    if err := _check_db("table listing"):
        return err
    result = db.list_tables()
    if result["success"]:
        tables = [row['table_name'] for row in result['data']]
        return f"Found {len(tables)} tables: {', '.join(tables)}"
    return f"Failed: {result['error']}"


@function_tool
def ping_database() -> str:
    """Test database connection status"""
    if err := _check_db("ping"):
        return err
    try:
        with db.get_connection() as conn:
            conn.execute(text("SELECT 1"))
        return "Database connection: OK"
    except SQLAlchemyError as e:
        return f"Database connection: FAILED - {e}"


def create_database_agent(model: str = "gpt-4o-mini") -> Agent:
    """Create database query agent with natural language interface"""
    return Agent(
        name="Database Query Agent",
        model=model,
        instructions="""PostgreSQL assistant. Tools: ping connection, list tables, get schemas, execute SELECT queries.
Process: List tables → Get relevant schemas → Build query → Execute → Present results.
Supports: JOINs, aggregations, GROUP BY, HAVING, subqueries, CTEs, ORDER BY, LIMIT.
Read-only: No INSERT/UPDATE/DELETE/DDL. Explain your approach and handle errors gracefully.""",
        functions=[ping_database, execute_sql_query, get_table_schema, list_all_tables],
    )


def main():
    """Interactive database query session"""
    if not db or not os.getenv("OPENAI_API_KEY"):
        print("\nError: Missing configuration.")
        print("Required in .env: DATABASE_URL=postgresql://user:pass@host:port/db")
        print("Required in .env: OPENAI_API_KEY=your_key")
        return

    agent = create_database_agent()
    print("\n" + "="*70)
    print("Database Query Agent | Ask in natural language | Type 'exit' to quit")
    print("="*70)

    while True:
        try:
            if not (user_input := input("\nYou: ").strip()):
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break

            print("Agent: ", end="", flush=True)
            print(asyncio.run(Runner.run(agent, user_input)).final_output)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")

    if db:
        db.close()


if __name__ == "__main__":
    main()
