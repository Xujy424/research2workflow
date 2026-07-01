"""Complete legacy analyzer/score/upgrade API, isolated from the new pipeline.

Heavy data-source and UI dependencies are imported only when their modules are
requested. The public methods from the original ``research_template`` scripts
are preserved in their original modules.
"""

__all__ = [
    "analyzer",
    "combination",
    "metrics",
    "portfolio",
    "score",
    "upgrade",
    "utils",
    "web",
]
