from __future__ import annotations

import unittest
from unittest import mock

from scripts.experiments.ucr_batch.run_when_gpu_idle import main, parse_gpu_ids, wait_for_gpu_idle


class RunWhenGpuIdleTest(unittest.TestCase):
    def test_parse_gpu_ids_validates_duplicates(self) -> None:
        self.assertEqual(parse_gpu_ids("0,2,5"), [0, 2, 5])
        with self.assertRaises(ValueError):
            parse_gpu_ids("0,2,2")

    def test_wait_for_gpu_idle_requires_consecutive_idle_polls(self) -> None:
        samples = iter(
            [
                {0: 6000, 1: 5500},
                {0: 800, 1: 700},
                {0: 900, 1: 650},
            ]
        )
        sleep_calls: list[float] = []
        log_messages: list[str] = []

        result = wait_for_gpu_idle(
            gpu_ids=[0, 1],
            memory_threshold_mb=1000,
            stable_polls=2,
            poll_interval_seconds=5.0,
            query_fn=lambda: next(samples),
            sleep_fn=sleep_calls.append,
            log_fn=log_messages.append,
        )

        self.assertEqual(result, {0: 900, 1: 650})
        self.assertEqual(sleep_calls, [5.0, 5.0])
        self.assertTrue(any("idle confirmation 2/2" in message for message in log_messages))

    def test_main_launches_follow_up_command(self) -> None:
        with (
            mock.patch("scripts.experiments.ucr_batch.run_when_gpu_idle.wait_for_gpu_idle", return_value={0: 400}),
            mock.patch("scripts.experiments.ucr_batch.run_when_gpu_idle.subprocess.run") as run_mock,
        ):
            run_mock.return_value = mock.Mock(returncode=0)
            exit_code = main(
                [
                    "--gpu-ids",
                    "0",
                    "--memory-threshold-mb",
                    "1000",
                    "--stable-polls",
                    "1",
                    "--poll-interval",
                    "1",
                    "--",
                    "echo",
                    "hello",
                ]
            )

        self.assertEqual(exit_code, 0)
        run_mock.assert_called_once_with(["echo", "hello"], cwd=None)


if __name__ == "__main__":
    unittest.main()
