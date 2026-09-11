"""Offline regressions for ranking, safe delivery, and the Streamlit journey."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import requests

from next_chapter.pipelines.linkedin_matching import match_linkedin_jobs
from next_chapter.delivery.telegram import create_results_telegram
from next_chapter.services.delivery_journal import UncertainDeliveryError, resolve_part


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.database = self.directory / "test.sqlite3"
        self.stack.enter_context(patch.dict(os.environ, {
            "AGENT2_DATABASE_PATH": str(self.database), "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "123", "PARSER_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "test-key", "COVER_LETTER_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key",
        }))

    def test_later_better_job_wins_and_bad_posting_does_not_consume_slot(self):
        jobs = [{"title": f"Role {i}", "company": "Example", "description": str(i),
                 "url": f"https://example.com/{i}"} for i in range(5)]
        def parse(job_description, **kwargs):
            if job_description == "1":
                raise ValueError("bad posting")
            return SimpleNamespace(required_skills=["Python"],
                required_experience_years=10 if job_description != "4" else 1)
        result = match_linkedin_jobs(SimpleNamespace(skills=["Python"], experience_years=1),
            query="Python", max_jobs=1, search_pool_size=5, search_fn=lambda **kwargs: jobs,
            parser_fn=parse)
        self.assertEqual(result["ranked_jobs"][0]["url"], "https://example.com/4")
        self.assertEqual(result["parsed_count"], 4)
        self.assertEqual(result["skipped_count"], 1)

    def telegram(self):
        return create_results_telegram(SimpleNamespace(), [], "letter", delivery_id="run-1",
                                       database_path=self.database)

    def test_partial_delivery_retries_only_failed_chunk(self):
        ok = Mock(status_code=200, json=lambda: {"ok": True, "result": {"message_id": 1}})
        failed = Mock(status_code=429, json=lambda: {"ok": False, "description": "retry later"})
        with patch("next_chapter.delivery.telegram._build_email_body", return_value="a\nb"), \
             patch("next_chapter.delivery.telegram._split_long_message", return_value=["a", "b"]), \
             patch("next_chapter.delivery.telegram.requests.post", side_effect=[ok, failed, ok]) as post:
            with self.assertRaises(RuntimeError):
                self.telegram()
            self.assertEqual(len(self.telegram()["messages"]), 2)
            self.assertEqual(post.call_count, 3)
            self.assertEqual([call.kwargs["json"]["text"] for call in post.call_args_list], ["a", "b", "b"])
            self.telegram()
            self.assertEqual(post.call_count, 3)

    def test_uncertain_delivery_needs_explicit_resolution(self):
        with patch("next_chapter.delivery.telegram._build_email_body", return_value="a"), \
             patch("next_chapter.delivery.telegram.requests.post", side_effect=requests.Timeout("secret-url")) as post:
            with self.assertRaises(UncertainDeliveryError) as caught:
                self.telegram()
            self.assertNotIn("secret-url", str(caught.exception))
            with self.assertRaises(UncertainDeliveryError):
                self.telegram()
            self.assertEqual(post.call_count, 1)
            resolve_part("telegram:run-1", 0, received=True, database_path=self.database)
            self.telegram()
            self.assertEqual(post.call_count, 1)

    def workflow_fakes(self):
        import next_chapter.agents.agent2 as workflow
        from next_chapter.storage.agent2_checkpointer import Agent2SqliteSaver
        from tests.support.agent2_fakes import _inject_workflow_fakes, _ManualMonkeyPatch
        patcher = _ManualMonkeyPatch()
        self.addCleanup(patcher.undo)
        _inject_workflow_fakes(patcher)
        graph = workflow._build_agent2_graph(Agent2SqliteSaver(self.database))
        self.stack.enter_context(patch.object(workflow, "_get_agent2_graph", return_value=graph))
        return workflow

    def test_progress_restore_edited_letter_and_duplicate_approval(self):
        workflow = self.workflow_fakes()
        from tests.support.agent2_fakes import _fake_cv
        updates = []
        result = workflow.run_agent2_full_auto(_fake_cv(), interactive_delivery=False,
            query="Junior Python", search_pool_size=8, on_progress=updates.append)
        self.assertEqual(result["query"], "Junior Python")
        self.assertIn("match_jobs", updates)
        self.assertEqual(workflow.get_agent2_workflow(result["workflow_id"])["status"], "awaiting_delivery")
        with patch("next_chapter.delivery.gmail.create_results_draft", return_value={"id": "draft"}) as send:
            completed = workflow.resume_agent2_workflow(result["workflow_id"], "gmail", cover_letter="Reviewed letter")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(send.call_args.args[2], "Reviewed letter")
            with self.assertRaises(ValueError):
                workflow.resume_agent2_workflow(result["workflow_id"], "gmail")
            self.assertEqual(send.call_count, 1)

    def test_delivery_retry_does_not_repeat_search_or_generation(self):
        workflow = self.workflow_fakes()
        from tests.support.agent2_fakes import _fake_cv
        with patch("next_chapter.delivery.telegram.create_results_telegram", side_effect=[
                RuntimeError("temporary rejection"), {"messages": [{"ok": True}]}]) as send:
            failed = workflow.run_agent2_full_auto(_fake_cv(), delivery_channel="telegram", interactive_delivery=False)
            self.assertEqual(failed["delivery"]["status"], "failed")
            with patch.object(workflow, "match_linkedin_jobs", side_effect=AssertionError("search repeated")):
                complete = workflow.retry_agent2_delivery(failed["workflow_id"])
            self.assertEqual(complete["status"], "completed")
            self.assertEqual(send.call_count, 2)

    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=30)

    @staticmethod
    def widget(items, label):
        return next(item for item in items if item.label == label)

    def assert_clean(self, app):
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual([e.value for e in app.error], [])

    def test_sample_navigation_and_profile_review(self):
        app = self.app().run()
        self.assert_clean(app)
        self.widget(app.button, "Explore a sample workspace").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["page"], "Job matches")
        self.assertEqual(len(app.metric), 6)
        self.widget(app.radio, "Workspace").set_value("Your profile").run()
        self.widget(app.button, "Save reviewed profile").click().run()
        self.assert_clean(app)
        self.assertTrue(app.session_state["profile_confirmed"])

    def test_streamlit_search_review_delivery(self):
        self.workflow_fakes()
        from next_chapter.ui.support import sample_profile
        app = self.app()
        app.session_state["profile"] = sample_profile()
        app.session_state["profile_confirmed"] = True
        app.session_state["page"] = "Job matches"
        app.run()
        self.widget(app.button, "Find my matches").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["result"]["status"], "awaiting_delivery")
        self.widget(app.text_area, "Review and edit your letter").set_value("My reviewed letter")
        self.widget(app.button, "Approve and deliver").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["result"]["cover_letter"], "My reviewed letter")
        self.assertEqual(app.session_state["result"]["status"], "completed")

    def test_tracker_status_notes_and_saved_profile(self):
        from next_chapter.ui.support import sample_profile, sample_result
        from next_chapter.services.application_tracker import save_application, get_application
        record = save_application(AgentProfile(sample_profile()), sample_result()["top_job"])
        app = self.app()
        app.session_state["page"] = "Applications"
        app.run()
        self.assert_clean(app)
        self.widget(app.selectbox, "Application status").set_value("applied")
        self.widget(app.text_area, "Add a note").set_value("Follow up next week")
        self.widget(app.button, "Save update").click().run()
        updated = get_application(record.application_id)
        self.assertEqual(updated.status, "applied")
        self.assertIn("Follow up next week", updated.notes)
        self.widget(app.button, "Use this saved profile for a new search").click().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state["page"], "Your profile")


def AgentProfile(value):
    from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo
    return Agent2CVInfo.model_validate(value)


if __name__ == "__main__":
    unittest.main()
