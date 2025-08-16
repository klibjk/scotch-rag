import subprocess

def test_agent_pipeline():
    result = subprocess.run([
        'python', 'src/agent.py', 'Show last 10 rows of Transactions and explain amount.'
    ], capture_output=True, text=True)
    output = result.stdout.lower()
    assert '--- answer table ---' in output
    assert '--- provenance ---' in output
    assert '--- llm answer ---' in output
    print('Single-sheet pipeline test passed!')

def test_agent_multisheet():
    result = subprocess.run([
        'python', 'src/agent.py', 'Show all expenses from the finance file'
    ], capture_output=True, text=True)
    output = result.stdout.lower()
    assert '--- querying file: finance | sheet: expenses' in output
    assert '--- answer table ---' in output
    assert 'expenses' in output or 'category' in output
    print('Multi-file, multi-sheet pipeline test passed!')

if __name__ == '__main__':
    test_agent_pipeline()
    test_agent_multisheet()
