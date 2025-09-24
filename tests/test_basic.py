"""
Basic tests that run without heavy ML dependencies
Tests core functionality and imports
"""

import pytest
import sys
import os

# Add parent directory to path for CI compatibility
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_python_version():
    """Test that Python version is compatible"""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"

def test_basic_imports():
    """Test that core modules can be imported"""
    try:
        # Test basic Python syntax of modules
        import ast

        # Test data.py syntax
        with open('src/data.py', 'r') as f:
            ast.parse(f.read())

        # Test model.py syntax
        with open('src/model.py', 'r') as f:
            ast.parse(f.read())

        # Test evaluation.py syntax
        with open('src/evaluation.py', 'r') as f:
            ast.parse(f.read())

        # Test monitoring.py syntax
        with open('src/monitoring.py', 'r') as f:
            ast.parse(f.read())

        print(" All modules have valid Python syntax")

    except SyntaxError as e:
        pytest.fail(f"Syntax error in modules: {e}")

def test_package_structure():
    """Test that package structure is correct"""

    # Check that __init__.py files exist
    assert os.path.exists('src/__init__.py'), "src/__init__.py missing"
    assert os.path.exists('tests/__init__.py'), "tests/__init__.py missing"

    # Check core modules exist
    core_modules = ['data.py', 'model.py', 'evaluation.py', 'monitoring.py', 'reporting.py']
    for module in core_modules:
        assert os.path.exists(f'src/{module}'), f"src/{module} missing"

def test_configuration_files():
    """Test that configuration files are valid"""

    # Test pyproject.toml
    assert os.path.exists('pyproject.toml'), "pyproject.toml missing"

    # Test requirements.txt
    assert os.path.exists('requirements.txt'), "requirements.txt missing"

    # Test that requirements.txt has content
    with open('requirements.txt', 'r') as f:
        requirements = f.read().strip()
        assert len(requirements) > 0, "requirements.txt is empty"
        assert 'scikit-learn' in requirements, "scikit-learn not in requirements"

def test_documentation_structure():
    """Test that documentation structure is correct"""

    # Check docs directory
    assert os.path.exists('docs/guides'), "docs/guides directory missing"

    # Check guide files exist
    guides = [
        'Complete_Beginners_Guide_to_EquiML.md',
        'Complete_Guide_to_Building_Fair_LLMs_with_EquiML.md',
        'Complete_Guide_to_Fine_Tuning_LLMs_with_LoRA_and_EquiML.md'
    ]

    for guide in guides:
        guide_path = f'docs/guides/{guide}'
        assert os.path.exists(guide_path), f"{guide} missing"

        # Check that guides have substantial content
        with open(guide_path, 'r') as f:
            content = f.read()
            assert len(content) > 10000, f"{guide} appears to be too short"

def test_examples_structure():
    """Test that examples directory is properly structured"""

    # Check examples directory
    assert os.path.exists('examples'), "examples directory missing"
    assert os.path.exists('examples/web_demo'), "examples/web_demo missing"
    assert os.path.exists('examples/web_demo/app.py'), "web_demo/app.py missing"

def test_github_files():
    """Test that GitHub-specific files exist"""

    github_files = [
        '.github/workflows/ci.yml',
        '.github/ISSUE_TEMPLATE.md',
        '.github/PULL_REQUEST_TEMPLATE.md',
        '.gitignore',
        'SECURITY.md'
    ]

    for file_path in github_files:
        assert os.path.exists(file_path), f"{file_path} missing"

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])