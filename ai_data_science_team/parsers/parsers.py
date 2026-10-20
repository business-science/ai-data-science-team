# BUSINESS SCIENCE UNIVERSITY
# AI DATA SCIENCE TEAM
# ***
# Parsers

from langchain_core.output_parsers import BaseOutputParser

import re

# Python Parser for output standardization  
class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # Ignore reasoning blocks before extracting generated code.
        text = re.sub(
            r"<think>.*?</think>",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Use the final Python code block in the remaining response.
        matches = re.findall(
            r"```python\s*(.*?)```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if matches:
            return matches[-1].strip()

        return text
# SQL Parser for output standardization  
class SQLOutputParser(BaseOutputParser):
    def parse(self, text: str):        
        def extract_sql_code(text):
            sql_code_match = re.search(r'```sql(.*?)```', text, re.DOTALL)
            sql_code_match_2 = re.search(r"SQLQuery:\s*(.*)", text)
            if sql_code_match:
                sql_code = sql_code_match.group(1).strip()
                return sql_code
            if sql_code_match_2:
                sql_code = sql_code_match_2.group(1).strip()
                return sql_code
            else:
                sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
                if sql_code_match:
                    sql_code = sql_code_match.group(1).strip()
                    return sql_code
                else:
                    return None
        sql_code = extract_sql_code(text)
        if sql_code is not None:
            return sql_code
        else:
            # Assume ```sql wasn't used
            return text
