"""
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "AI Elder Care Team"
__description__ = "Multimodal loneliness detection system for elderly care"

from . import features
from . import models 
from . import explainability
from . import utils

__all__ = ['features', 'models', 'explainability', 'utils']
