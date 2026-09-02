"""Compatibility facade for Agent 2 parsing.

Agent 2's CV and LinkedIn job parsers are implemented separately. This module
keeps the original import path stable for the pipeline, development utilities,
and existing tests.
"""

from core.agent2_parser_common import (
    Agent2ParserError,
    CV_CACHE_VERSION,
    JOB_CACHE_VERSION,
    MAX_CV_SKILLS,
    MAX_JOB_SKILLS,
    MAX_RAW_CV_SKILLS,
    MAX_RAW_JOB_SKILLS,
    MIN_JOB_DESCRIPTION_CHARACTERS,
)
from core.agent2_cv_parser import (
    Agent2CVInfo,
    _looks_like_contextual_cv_skill,
    extract_cv_info_agent2,
)
from core.agent2_job_parser import (
    Agent2JobRequirements,
    _required_alternative_groups,
    extract_job_requirements_agent2,
)


__all__ = [
    "Agent2CVInfo",
    "Agent2JobRequirements",
    "Agent2ParserError",
    "CV_CACHE_VERSION",
    "JOB_CACHE_VERSION",
    "MAX_CV_SKILLS",
    "MAX_JOB_SKILLS",
    "MAX_RAW_CV_SKILLS",
    "MAX_RAW_JOB_SKILLS",
    "MIN_JOB_DESCRIPTION_CHARACTERS",
    "extract_cv_info_agent2",
    "extract_job_requirements_agent2",
]

