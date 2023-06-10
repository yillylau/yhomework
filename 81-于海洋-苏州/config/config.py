#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@date 2023/5/24
@author: y.haiyang@outlook.com

"""
from pathlib import Path


class Config:
    RES_PATH = Path(Path(__file__).parent.parent, "res")

    LENNA = str(Path(RES_PATH, "lenna.png"))
