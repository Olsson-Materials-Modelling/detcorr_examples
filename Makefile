.PHONY: regenerate manifests test verify verify-release case-studies-test \
	case-studies-verify-reference verify-all verify-all-release

PYTHON ?= python

regenerate:
	$(PYTHON) examples/run_worked_examples.py

manifests:
	$(PYTHON) examples/build_release_manifests.py

test:
	$(PYTHON) -m pytest examples --require-detcorr

verify:
	$(PYTHON) examples/verify_examples.py --require-detcorr --run-worked-examples

verify-release:
	$(PYTHON) examples/verify_examples.py --require-detcorr --run-worked-examples \
		--require-byte-identical-outputs

case-studies-test:
	$(MAKE) -C case_studies PYTHON="$(PYTHON)" test

case-studies-verify-reference:
	$(MAKE) -C case_studies PYTHON="$(PYTHON)" verify-reference

verify-all: verify case-studies-test case-studies-verify-reference

verify-all-release: verify-release case-studies-test case-studies-verify-reference
