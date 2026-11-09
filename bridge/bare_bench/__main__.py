"""`python -m bare_bench run --config ... --output ...` 的入口。"""
import sys

from bare_bench.runner import main

if __name__ == "__main__":
    sys.exit(main())
