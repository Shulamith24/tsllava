from __future__ import annotations

import unittest

from scripts.experiments.ucr_batch.run_ucr_experiment_queue import (
    build_command,
    format_job_name,
    parse_args,
)


class UCRExperimentQueueTest(unittest.TestCase):
    def test_dataset_families_expand_job_template_and_forward_family(self) -> None:
        args, forward_args = parse_args(
            [
                "--experiments",
                "m2_pretrained patchtst",
                "--protocol",
                "fewshot",
                "--dataset-families",
                "cinc2017af,cinc2016heart",
                "--gpu-ids",
                "0",
                "--shots",
                "1",
            ]
        )

        self.assertEqual(args.dataset_families, ["cinc2017af", "cinc2016heart"])
        self.assertEqual(args.job_name_template, "{dataset_family}_{experiment}")
        self.assertEqual(forward_args, ["--shots", "1"])

        job_name = format_job_name(
            args.job_name_template,
            experiment="m2_pretrained",
            protocol="fewshot",
            queue_index=1,
            dataset_family="cinc2017af",
        )
        command = build_command(
            experiment="m2_pretrained",
            protocol="fewshot",
            job_name=job_name,
            data_path="./data",
            gpu_id="0",
            dataset_family="cinc2017af",
            forward_args=forward_args,
            dry_run=True,
        )

        self.assertEqual(job_name, "cinc2017af_m2_pretrained")
        self.assertIn("--dry-run", command)
        self.assertIn("--dataset_family", command)
        self.assertIn("cinc2017af", command)
        self.assertIn("--shots", command)
        self.assertIn("1", command)


if __name__ == "__main__":
    unittest.main()
