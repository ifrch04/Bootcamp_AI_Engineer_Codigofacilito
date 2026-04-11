.PHONY: test-contracts test-goldens eval demo-clase14 clean-results

test-contracts:
	python -m pytest tests/test_contracts.py -v

test-goldens:
	python -m pytest tests/test_goldens.py -v --tb=short

eval:
	python -m evals.runner

demo-clase14:
	python -m demos.clase14_demo

clean-results:
	rm -rf evals/results/*.json
