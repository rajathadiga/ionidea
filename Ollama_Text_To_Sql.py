from typing import Dict, Any
from sqlalchemy import create_engine, inspect, text
import time
from llama_index.llms.ollama import Ollama
import pandas as pd

class TextToSQL:
    def __init__(self, connection_string: str, model: str = "llama3-groq-tool-use:latest"):
        """
        Initialize the Text-to-SQL converter using Ollama
        
        Args:
            connection_string: SQLAlchemy connection string for the database
            Examples:
            - SQLite: "sqlite:///database.db"
            - PostgreSQL: "postgresql://user:password@localhost:5432/dbname"
            - MySQL/MariaDB: "mysql+pymysql://user:password@localhost:3306/dbname"
            model: Ollama model to use (default: "llama3.2:latest")
        """
        self.connection_string = connection_string
        self.model = model
        
        # Create SQLAlchemy engine
        self.engine = create_engine(connection_string)
        
        # Test connection
        try:
            with self.engine.connect() as conn:
                pass
        except Exception as e:
            raise ConnectionError(f"Could not connect to database: {str(e)}")
        
        # Initialize Ollama client
        self._init_ollama()
        
        # Cache for schema info
        self._schema_cache = None
        self._schema_cache_timestamp = 0
        self._schema_cache_ttl = 300  # 5 minutes TTL for schema cache
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            self.ollama_client = Ollama(
                model=self.model,
                request_timeout=60.0,
                temperature=0.2
            )
        except Exception as e:
            raise ConnectionError(f"Could not initialize Ollama client: {str(e)}")
    
    def _generate_sql(self, query: str, schema: str) -> str:
        """
        Generate SQL from natural language using Ollama.
        """
        # Prepare the prompt
        prompt = (
            "You are a SQL expert. Given an input question, create a syntactically correct SQL query to run.\n"
            "Use the schema below to create your query:\n"
            f"{schema}\n\n"
            "IMPORTANT: Return ONLY the SQL query without any explanations, comments, or markdown formatting and never give ```sql\n or.\n"
            "Follow these rules to make your query more useful:\n"
            "1. Always include primary key/ID columns in SELECT statements\n"
            "2. Include descriptive columns that help identify the records (names, titles, dates, etc.)\n"
            "3. For aggregate queries, include grouping fields in the result\n"
            "4. When filtering or searching, return enough context columns for the results to be meaningful\n"
            "5. Limit results only when explicitly asked, otherwise return complete datasets\n"
            "6. Use appropriate joins to include related information when relevant\n"
            "7. IMPORTANT: When using table aliases, always verify each column belongs to the correct table\n"
            "8. Do not reference columns that don't exist in the tables - check the schema carefully\n"
            "9. Prefer using descriptive table aliases (like 'products p' instead of 'products T1')\n"
            "10. If the query cannot be answered, return 'No results found'\n"
            f"Make sure to use SQL syntax compatible with {self.engine.dialect.name}.\n"
            "You are mostly working with Job desctriptions, applicants, interviewers, and companies.\n"
            "Make use of interviewrs only if required of interviewr details are required.\n"
            "You are not allowed to use any other tables.\n"
            f"Input question: {query}\n\n"
            "SQL query: "
        )

        
        # Call Ollama
        try:
            response = self.ollama_client.complete(prompt, temperature=0)
            sql_query = response.text.strip()
            return sql_query
        except Exception as e:
            print(f"Error generating SQL: {str(e)}")
            return ""
    
    def get_table_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive information about all tables and their schema from the database.
        Works with any RDBMS supported by SQLAlchemy (SQLite, MySQL, PostgreSQL, Oracle, MS SQL, etc.).
        
        Returns:
            Dict with table information in the format:
            {
                "table_name": {
                    "columns": {
                        "column_name": {
                            "type": str,
                            "nullable": bool,
                            "primary_key": bool,
                            "default": value,
                            "comment": str
                        }
                    },
                    "indexes": [...],
                    "foreign_keys": [...]
                }
            }
        """
        # Check if we have a valid cached schema
        current_time = time.time()
        if self._schema_cache and (current_time - self._schema_cache_timestamp) < self._schema_cache_ttl:
            return self._schema_cache
            
        tables = {}
        
        try:
            # Get dialect name for specialized handling
            dialect_name = self.engine.dialect.name.lower()            
            # Create SQLAlchemy inspector - this works for all database types
            inspector = inspect(self.engine)
            
            # Get all table names
            try:
                table_names = inspector.get_table_names()
            except Exception as e:
                table_names = []
            
            # Empty database case
            if not table_names:
                self._schema_cache = {}
                self._schema_cache_timestamp = current_time
                return {}
            
            # Process all tables
            for table_name in table_names:
                tables[table_name] = {"columns": {}, "indexes": [], "foreign_keys": []}
                
                # Get primary key info
                try:
                    pk_info = inspector.get_pk_constraint(table_name)
                    pk_columns = pk_info.get('constrained_columns', []) if pk_info else []
                except Exception as e:
                    pk_columns = []
                
                # Get all columns
                try:
                    columns = inspector.get_columns(table_name)
                except Exception as e:
                    columns = []
                
                # Process column information
                for column in columns:
                    col_name = column['name']
                    col_type = str(column['type'])
                    nullable = column.get('nullable', True)
                    default = column.get('default', None)
                    comment = column.get('comment', '')
                    is_pk = col_name in pk_columns
                    
                    tables[table_name]["columns"][col_name] = {
                        "type": col_type,
                        "nullable": nullable,
                        "primary_key": is_pk,
                        "default": default,
                        "comment": comment
                    }
                
                # Get foreign key constraints if available
                try:
                    fk_constraints = inspector.get_foreign_keys(table_name)
                    tables[table_name]["foreign_keys"] = fk_constraints
                except Exception as e:
                    print(f"  Error getting foreign keys for {table_name}: {str(e)}")
                
                # Get index information if available
                try:
                    indexes = inspector.get_indexes(table_name)
                    tables[table_name]["indexes"] = indexes
                except Exception as e:
                    print(f"  Error getting indexes for {table_name}: {str(e)}")
            
            # For specific dialects, enhance with additional metadata if needed
            if "sqlite" in dialect_name:
                # SQLite-specific enhancements
                try:
                    with self.engine.connect() as conn:
                        for table_name in table_names:
                            # Get additional table metadata from SQLite
                            result = conn.execute(text(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
                            table_sql = result.scalar()
                            if table_sql:
                                tables[table_name]["creation_sql"] = table_sql
                except Exception as e:
                    print(f"Error getting additional SQLite metadata: {str(e)}")
            
            elif "mysql" in dialect_name or "maria" in dialect_name:
                # MySQL/MariaDB-specific enhancements
                try:
                    with self.engine.connect() as conn:
                        database_name = conn.execute(text("SELECT DATABASE()")).scalar()
                        if database_name:
                            for table_name in table_names:
                                # Get table comment
                                result = conn.execute(text(
                                    f"SELECT table_comment FROM information_schema.tables "
                                    f"WHERE table_schema = '{database_name}' AND table_name = '{table_name}'"
                                ))
                                table_comment = result.scalar()
                                if table_comment:
                                    tables[table_name]["comment"] = table_comment
                except Exception as e:
                    print(f"Error getting additional MySQL/MariaDB metadata: {str(e)}")
            
            elif "postgre" in dialect_name:
                # PostgreSQL-specific enhancements
                try:
                    with self.engine.connect() as conn:
                        for table_name in table_names:
                            # Get table comments
                            schema_name = 'public'  # Default schema
                            result = conn.execute(text(
                                f"SELECT obj_description('{schema_name}.{table_name}'::regclass, 'pg_class')"
                            ))
                            table_comment = result.scalar()
                            if table_comment:
                                tables[table_name]["comment"] = table_comment
                except Exception as e:
                    print(f"Error getting additional PostgreSQL metadata: {str(e)}")
        
        except Exception as e:
            print(f"Fatal error in get_table_info: {str(e)}")
            # Return empty dict in case of error, but don't cache it
            return {}
        
        # Update the cache
        self._schema_cache = tables
        self._schema_cache_timestamp = current_time
        
        return tables
    
    def _format_schema_for_prompt(self) -> str:
        """
        Format the database schema in a concise way for the LLM prompt.
        Includes tables, columns, data types, constraints, and relationships.
        """
        tables = self.get_table_info()
        schema_str = []
        
        if not tables:
            return "No tables found in the database."
        
        for table_name, table_info in tables.items():
            # Format columns
            columns = []
            for col_name, col_info in table_info["columns"].items():
                # Format type information and constraints
                pk_str = " PRIMARY KEY" if col_info["primary_key"] else ""
                null_str = " NOT NULL" if not col_info["nullable"] else ""
                
                # Add default if present
                default_str = ""
                if col_info["default"] is not None and str(col_info["default"]) != "None":
                    default_value = str(col_info["default"])
                    default_str = f" DEFAULT {default_value}"
                
                # Add comment if present
                comment_str = ""
                if col_info.get("comment") and col_info["comment"].strip():
                    comment_str = f" -- {col_info['comment']}"
                
                columns.append(f"  {col_name} {col_info['type']}{pk_str}{null_str}{default_str}{comment_str}")
            
            # Format foreign keys if available
            if "foreign_keys" in table_info and table_info["foreign_keys"]:
                for fk in table_info["foreign_keys"]:
                    if "constrained_columns" in fk and "referred_table" in fk and "referred_columns" in fk:
                        fk_cols = ", ".join(fk["constrained_columns"])
                        ref_cols = ", ".join(fk["referred_columns"])
                        ref_table = fk["referred_table"]
                        
                        columns.append(f"  FOREIGN KEY ({fk_cols}) REFERENCES {ref_table}({ref_cols})")
            
            # Assemble CREATE TABLE statement
            table_comment = f" -- {table_info.get('comment')}" if table_info.get("comment") else ""
            schema_str.append(f"CREATE TABLE {table_name} ({table_comment}\n" + ",\n".join(columns) + "\n);")
            
            # Add index information if available
            if "indexes" in table_info and table_info["indexes"]:
                for idx in table_info["indexes"]:
                    if "name" in idx and "column_names" in idx:
                        idx_type = "UNIQUE " if idx.get("unique", False) else ""
                        idx_cols = ", ".join(idx["column_names"])
                        idx_name = idx["name"]
                        
                        schema_str.append(f"CREATE {idx_type}INDEX {idx_name} ON {table_name} ({idx_cols});")
        
        return "\n\n".join(schema_str)
    
    def natural_language_to_sql(self, query: str) -> Dict[str, Any]:
        """
        Use only natural, simple English. Tell me exactly what you want.
        Ask about job openings, interview schedules, or company information.
        """
        try:
            # Format the schema for the prompt
            print("I got this:", query)
            schema = self._format_schema_for_prompt()            
            # Generate SQL with Ollama
            sql_query = self._generate_sql(query, schema)
            print(f"Generated SQL query: {sql_query}")
            
            if not sql_query or sql_query.strip() == "":
                return {
                    "error": "Failed to generate SQL query"
                }
            
            if sql_query.lower().strip() == "no results found":
                return {
                    "error": "The query could not be answered with the available schema"
                }
            
            # Execute the query
            try:
                df = pd.read_sql(sql_query, self.engine)
                print(f"Query executed successfully. Results shape: {df.shape}")
                
                # Convert DataFrame to a complete representation
                if len(df) > 0:
                    # Format as a list of dictionaries for complete data
                    results_list = df.to_dict(orient='records')
                    
                    # Limit to a reasonable number of records if too large
                    max_records = 20
                    if len(results_list) > max_records:
                        truncated_results = results_list[:max_records]
                        result_str = f"Found {len(results_list)} results. Showing first {max_records}:\n\n"
                        result_str += self._format_results(truncated_results)
                    else:
                        result_str = f"Found {len(results_list)} results:\n\n"
                        result_str += self._format_results(results_list)
                else:
                    result_str = "No results found for your query."
                
                return {
                    "results": result_str,
                }
            except Exception as e:
                error_msg = f"Error executing query: {str(e)}"
                print(f"❌ {error_msg}")
                return {
                    "error": error_msg
                }
        except Exception as e:
            error_msg = f"Error in natural language to SQL conversion: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "error": error_msg
            }
    
    def _format_results(self, results_list):
        """Format a list of dictionaries into a readable string."""
        if not results_list:
            return "No results found."
            
        # Get column widths
        columns = list(results_list[0].keys())
        col_widths = {col: max(len(str(col)), max(len(str(row[col])) for row in results_list)) for col in columns}
        
        # Format header
        header = " | ".join(f"{col:{col_widths[col]}}" for col in columns)
        separator = "-" * len(header)
        
        # Format rows
        rows = []
        for row in results_list:
            formatted_row = " | ".join(f"{str(row[col]):{col_widths[col]}}" for col in columns)
            rows.append(formatted_row)
        
        # Combine everything
        result = f"{header}\n{separator}\n" + "\n".join(rows)
        return result

def main():
    """Demo function to test the Text-to-SQL converter."""
    # Example connection strings
    connections = {
        "sqlite": "sqlite:///ai_recruitment.sqlite",
    }
    import os
    print(os.path.exists("ai_recruitment.sqlite"))

    # Use SQLite for demo
    text_to_sql = TextToSQL(connections["sqlite"])
    
    # Print database schema information
    tables = text_to_sql.get_table_info()
    print("Database Schema:")
    for table_name, table_info in tables.items():
        print(f"Table: {table_name}")
        for column, details in table_info['columns'].items():
            pk = " (PK)" if details['primary_key'] else ""
            null = "" if details['nullable'] else " NOT NULL"
            print(f"  - {column}: {details['type']}{pk}{null}")
    
    # Sample queries to test
    test_queries = [
        "Which companies are currently hiring for an AI-related role?",
        "List all applicants along with a short snippet of their resume.",
        "Which interviewer is assigned to each job description, and what is their position?",
    ]
    
    # Test each query
    print("\nTesting queries:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}]: {query}")
        
        result = text_to_sql.natural_language_to_sql(query)
        
        if not result.get('error'):
            print("Results : ",result['results'])
        else:
            print(f"Error: {result['error']}")
            

if __name__ == "__main__":
    main() 