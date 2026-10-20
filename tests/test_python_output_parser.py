import unittest

from ai_data_science_team.parsers.parsers import PythonOutputParser


class TestPythonOutputParser(unittest.TestCase):
    def setUp(self):
        self.parser = PythonOutputParser()

    def test_ignores_python_code_inside_think_block(self):
        text = """<think>
```python
wrong = True
```
</think>
```python
right = True
```"""

        self.assertEqual(self.parser.parse(text), "right = True")

    def test_selects_final_python_code_block(self):
        text = """```python
first = 1
```
```python
second = 2
```"""

        self.assertEqual(self.parser.parse(text), "second = 2")

    def test_preserves_plain_text_without_python_block(self):
        text = "No code block was provided."

        self.assertEqual(self.parser.parse(text), text)


if __name__ == "__main__":
    unittest.main()
