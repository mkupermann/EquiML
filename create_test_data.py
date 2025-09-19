import pandas as pd
import numpy as np

# Create a dummy DataFrame
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [1.1, 2.2, 3.3],
    'c': ['foo', 'bar', 'baz'],
    'd': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
})

# Save as Parquet
df.to_parquet('tests/data.parquet')

# Save as CSV
df.to_csv('tests/data.csv', index=False)

# Save as JSON
df.to_json('tests/data.json', orient='records')

# Save as Excel
df.to_excel('tests/data.xlsx', index=False)


# Save as ARFF manually
arff_content = """
@RELATION data
@ATTRIBUTE a INTEGER
@ATTRIBUTE b REAL
@ATTRIBUTE c STRING
@ATTRIBUTE d STRING
@DATA
1,1.1,"'foo'","'2024-01-01 00:00:00'"
2,2.2,"'bar'","'2024-01-02 00:00:00'"
3,3.3,"'baz'","'2024-01-03 00:00:00'"
"""

with open('tests/data.arff', 'w') as f:
    f.write(arff_content)

print("Test data files created successfully.")
