import unittest

from ai_data_science_team.agents.data_wrangling_agent import (
    _validate_data_wrangler_result,
)


class TestDataWranglerResultValidation(unittest.TestCase):
    def test_none_result_is_an_error(self):
        error = _validate_data_wrangler_result(None, None)

        self.assertEqual(
            error,
            "Data wrangling function returned None instead of a table.",
        )

    def test_empty_dictionary_is_valid(self):
        self.assertIsNone(_validate_data_wrangler_result({}, None))

    def test_execution_error_is_preserved(self):
        self.assertEqual(
            _validate_data_wrangler_result(None, "sandbox failed"),
            "sandbox failed",
        )


if __name__ == "__main__":
    unittest.main()
