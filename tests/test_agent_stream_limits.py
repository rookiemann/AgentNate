import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from backend.routes.tools import _clamp_max_tool_calls, MIN_AGENT_TOOL_CALLS, MAX_AGENT_TOOL_CALLS


class TestAgentStreamLimits(unittest.TestCase):
    def test_clamp_defaults(self):
        self.assertEqual(_clamp_max_tool_calls(None), 25)

    def test_clamp_lower_bound(self):
        self.assertEqual(_clamp_max_tool_calls(0), MIN_AGENT_TOOL_CALLS)
        self.assertEqual(_clamp_max_tool_calls(-99), MIN_AGENT_TOOL_CALLS)

    def test_clamp_upper_bound(self):
        self.assertEqual(_clamp_max_tool_calls(MAX_AGENT_TOOL_CALLS + 1), MAX_AGENT_TOOL_CALLS)
        self.assertEqual(_clamp_max_tool_calls(9999), MAX_AGENT_TOOL_CALLS)

    def test_clamp_in_range(self):
        self.assertEqual(_clamp_max_tool_calls(1), 1)
        self.assertEqual(_clamp_max_tool_calls(25), 25)
        self.assertEqual(_clamp_max_tool_calls(50), 50)


if __name__ == "__main__":
    unittest.main()
