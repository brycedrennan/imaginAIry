"""Pythonic AI generation of images and videos"""
import os

# tells pytorch to allow MPS usage (for Mac M1 compatibility)
os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# use more memory than we should
os.putenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
