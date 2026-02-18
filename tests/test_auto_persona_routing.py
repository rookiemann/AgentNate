import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.routes.tools import _select_auto_persona


class _DummyPersonaManager:
    def __init__(self, available):
        self.available = set(available)

    def get(self, persona_id):
        return object() if persona_id in self.available else None


class TestAutoPersonaRouting(unittest.TestCase):
    def setUp(self):
        self.pm = _DummyPersonaManager(
            [
                "general_assistant",
                "researcher",
                "coder",
                "automator",
                "image_creator",
                "system_agent",
            ]
        )

    def test_greeting_routes_to_general_assistant(self):
        persona = _select_auto_persona("hey thanks", self.pm)
        self.assertEqual(persona, "general_assistant")

    def test_research_routes_to_researcher(self):
        persona = _select_auto_persona(
            "Research the latest AI news and summarize sources.",
            self.pm,
        )
        self.assertEqual(persona, "researcher")

    def test_code_routes_to_coder(self):
        persona = _select_auto_persona(
            "Debug this python function causing a TypeError in class method.",
            self.pm,
        )
        self.assertEqual(persona, "coder")

    def test_workflow_routes_to_automator(self):
        persona = _select_auto_persona(
            "Build an n8n workflow with webhook trigger and schedule.",
            self.pm,
        )
        self.assertEqual(persona, "automator")

    def test_image_routes_to_image_creator(self):
        persona = _select_auto_persona(
            "Create a comfy SDXL image with lora checkpoint.",
            self.pm,
        )
        self.assertEqual(persona, "image_creator")

    def test_ambiguous_mixed_intent_falls_back_system_agent(self):
        persona = _select_auto_persona(
            "Write code and then create a workflow.",
            self.pm,
        )
        self.assertEqual(persona, "system_agent")

    def test_unavailable_selected_persona_falls_back_system_agent(self):
        pm = _DummyPersonaManager(["system_agent", "researcher"])
        persona = _select_auto_persona("Create comfy image with flux model", pm)
        self.assertEqual(persona, "system_agent")


if __name__ == "__main__":
    unittest.main()
