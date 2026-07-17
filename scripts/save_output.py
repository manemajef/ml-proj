from io import StringIO
import re
from bs4 import BeautifulSoup
import pandas as pd

# 1. Configuration: Change to your actual file name
input_file = "docs/notebook.md"
# input_file = "docs/Group_27_Notebook.md"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 2. Regex to find all <div> blocks containing dataframe tables
# This handles multiline blocks and makes the <style> block optional
div_pattern = re.compile(
    r"<div>\s*(?:<style.*?>.*?</style>\s*)?<table[^>]*?class=\"[^\"]*dataframe[^\"]*\".*?>.*?</table>\s*</div>",
    re.DOTALL,
)


def html_to_md_table(match):
    html_block = match.group(0)

    # Extract the table using BeautifulSoup
    soup = BeautifulSoup(html_block, "html.parser")
    table_html = soup.find("table")

    if not table_html:
        return html_block  # Return unchanged if table extraction fails

    try:
        # Convert HTML table to Pandas DataFrame
        df_list = pd.read_html(StringIO(str(table_html)))
        if not df_list:
            return html_block

        df = df_list[0]

        # Simplify MultiIndex columns if present (e.g., pivot tables or complex headers)
        if isinstance(df.columns, pd.MultiIndex):
            new_columns = []
            for col_tuple in df.columns:
                parts = [str(x) for x in col_tuple if not str(x).startswith("Unnamed:")]
                new_columns.append(" ".join(parts))
            df.columns = new_columns

        # If the first column is completely unnamed (Pandas index), make it the index
        has_unnamed_index = False
        if str(df.columns[0]).startswith("Unnamed:"):
            df = df.set_index(df.columns[0])
            df.index.name = ""  # Clean up index label
            has_unnamed_index = True

        # Optimize float columns for readability (round to 4 decimal places)
        for col in df.select_dtypes(include=["float"]):
            df[col] = df[col].round(4)

        # Convert to native markdown table syntax
        # Only include the index if we converted an unnamed first column back to the index
        return "\n\n" + df.to_markdown(index=has_unnamed_index) + "\n\n"
    except Exception as e:
        # Fallback in case of parsing errors
        print(f"Skipping a table due to error: {e}")
        return html_block


# 3. Replace all HTML dataframe tables with markdown tables
cleaned_content = div_pattern.sub(html_to_md_table, content)

# 4. Strip angle brackets from image and standard markdown links (e.g. ![svg](<path>) -> ![svg](path))
# GitHub flavored markdown does not render image links with angle brackets properly
cleaned_content, sub_count = re.subn(
    r"(!?\[[^\]]*\])\(<([^>]+)>\)", r"\1(\2)", cleaned_content
)
print(f"DEBUG: Number of link substitutions made: {sub_count}")

# 5. Save the cleaned markdown back
with open(input_file, "w", encoding="utf-8") as f:
    f.write(cleaned_content)

print(f"Successfully cleaned HTML tables and image/markdown links in '{input_file}'!")

# DEBUG: read it back
with open(input_file, "r", encoding="utf-8") as f:
    debug_text = f.read()
print("DEBUG: contains angle brackets after write:", "<notebook_files" in debug_text)
