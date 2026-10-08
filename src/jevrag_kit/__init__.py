"""jevrag-kit: a configurable JEV passage classifier, LLM layer, and JEV claims checker.

    from jevrag_kit import Engine, Passage, load_config, load_env_file

    load_env_file(".env")                                  # optional: keys from a .env file
    engine = Engine.from_config(load_config("jev.yaml"))   # keys from the environment
    result = engine.run(query, passages)                   # passages in retrieval order
    print(result.answer.text)                              # store result.trace for audit and tuning

The stages are also usable on their own: jevrag_kit.classifier, jevrag_kit.llm, jevrag_kit.checker.
"""

from jevrag_kit._version import __version__
from jevrag_kit.config import JevConfig, load_config, parse_config
from jevrag_kit.engine import Classification, Engine, RunResult
from jevrag_kit.envfile import load_env_file
from jevrag_kit.errors import ConfigError, JevragKitError, MissingCredentials, MissingDependency
from jevrag_kit.factory import build_engine, build_generator, build_scorer, build_verifier
from jevrag_kit.text import normalize
from jevrag_kit.types import Answer, AnswerClaim, Claim, Draft, Passage, PassageScores, RelationResult

__all__ = [
    "Answer",
    "AnswerClaim",
    "Claim",
    "Classification",
    "ConfigError",
    "Draft",
    "Engine",
    "JevConfig",
    "JevragKitError",
    "MissingCredentials",
    "MissingDependency",
    "Passage",
    "PassageScores",
    "RelationResult",
    "RunResult",
    "__version__",
    "build_engine",
    "build_generator",
    "build_scorer",
    "build_verifier",
    "load_config",
    "load_env_file",
    "normalize",
    "parse_config",
]
