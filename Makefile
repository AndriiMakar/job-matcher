# Local dev shortcuts for the eval harness.
# Override the connection if your Postgres differs:  make eval TEST_DB=postgresql://...
TEST_DB ?= postgresql://postgres:postgres@localhost:5433/jobmatcher_test
GOLD    ?= tests/fixtures/gold_labels.csv

.PHONY: seed-test-db eval

## seed-test-db: (re)load the fixed 8-job corpus into the throwaway test DB. Run once.
seed-test-db:
	DATABASE_URL="$(TEST_DB)" ALLOW_SEED_OVERWRITE=1 python scripts/seed_test_jobs.py

## eval: run the CI eval gate locally (recall + judge agreement) against the test DB + fixture.
eval:
	DATABASE_URL="$(TEST_DB)" GOLD_LABELS="$(GOLD)" python -m pytest tests/ -v
