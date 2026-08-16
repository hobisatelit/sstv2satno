#!/usr/bin/env python3
# forked from: https://github.com/colaclanth/sstv
# Copyright 2026 colaclanth, hobisatelit
# https://github.com/hobisatelit/sstv2satno
# License: GPL-3.0-or-later

VERSION = "0.1-ENHANCED-GENERAL"

import sys
import os
import signal
import wave
from enum import Enum
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.signal.windows import hann
from os import get_terminal_size
from sys import stderr, stdout, platform, argv, exit

"""Constants for SSTV specification and each supported mode"""
class COL_FMT(Enum):
    RGB = 1
    GBR = 2
    YUV = 3
    BW = 4

class M1(object):
    NAME = "Martin 1"
    COLOR = COL_FMT.GBR
    LINE_WIDTH = 320
    LINE_COUNT = 256
    SYNC_PULSE = 0.004862
    SYNC_PORCH = 0.000572256
    SCAN_TIME = 0.146432
    SEP_PULSE = 0.000572
    CHAN_COUNT = 3
    CHAN_SYNC = 0
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME)
    CHAN_OFFSETS.append(CHAN_OFFSETS[1] + CHAN_TIME)
    LINE_TIME = SYNC_PULSE + SYNC_PORCH + 3 * CHAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 2.34  
    HAS_START_SYNC = False
    HAS_HALF_SCAN = False
    HAS_ALT_SCAN = False

class M2(M1):
    NAME = "Martin 2"
    LINE_WIDTH = 320
    SCAN_TIME = 0.073216
    SYNC_PULSE = 0.004862
    SYNC_PORCH = 0.000572
    SEP_PULSE = 0.000572
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME)
    CHAN_OFFSETS.append(CHAN_OFFSETS[1] + CHAN_TIME)
    LINE_TIME = SYNC_PULSE + SYNC_PORCH + 3 * CHAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 4.68

class S1(object):
    NAME = "Scottie 1"
    COLOR = COL_FMT.GBR
    LINE_WIDTH = 320
    LINE_COUNT = 256
    SCAN_TIME = 0.138240
    SYNC_PULSE = 0.009000
    SYNC_PORCH = 0.001500
    SEP_PULSE = 0.001500
    CHAN_COUNT = 3
    CHAN_SYNC = 2
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH + CHAN_TIME]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME)
    CHAN_OFFSETS.append(SYNC_PULSE + SYNC_PORCH)
    LINE_TIME = SYNC_PULSE + 3 * CHAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 2.48
    HAS_START_SYNC = True
    HAS_HALF_SCAN = False
    HAS_ALT_SCAN = False

class S2(S1):
    NAME = "Scottie 2"
    LINE_WIDTH = 320
    SCAN_TIME = 0.088064
    SYNC_PULSE = 0.009000
    SYNC_PORCH = 0.001500
    SEP_PULSE = 0.001500
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH + CHAN_TIME]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME)
    CHAN_OFFSETS.append(SYNC_PULSE + SYNC_PORCH)
    LINE_TIME = SYNC_PULSE + 3 * CHAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 3.82

class SDX(S2):
    NAME = "Scottie DX"
    LINE_WIDTH = 320
    SCAN_TIME = 0.345600
    SYNC_PULSE = 0.009000
    SYNC_PORCH = 0.001500
    SEP_PULSE = 0.001500
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH + CHAN_TIME]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME)
    CHAN_OFFSETS.append(SYNC_PULSE + SYNC_PORCH)
    LINE_TIME = SYNC_PULSE + 3 * CHAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 0.98

class R36(object):
    NAME = "Robot 36"
    COLOR = COL_FMT.YUV
    LINE_WIDTH = 320
    LINE_COUNT = 240
    SCAN_TIME = 0.088000
    HALF_SCAN_TIME = 0.044000
    SYNC_PULSE = 0.009000
    SYNC_PORCH = 0.003000
    SEP_PULSE = 0.004500
    SEP_PORCH = 0.001500
    CHAN_COUNT = 2
    CHAN_SYNC = 0
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME + SEP_PORCH)
    LINE_TIME = CHAN_OFFSETS[1] + HALF_SCAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    HALF_PIXEL_TIME = HALF_SCAN_TIME / LINE_WIDTH
    #WINDOW_FACTOR = 7.70
    WINDOW_FACTOR = 7.85
    HAS_START_SYNC = False
    HAS_HALF_SCAN = True
    HAS_ALT_SCAN = True

class R72(R36):
    NAME = "Robot 72"
    LINE_WIDTH = 320
    SCAN_TIME = 0.138000
    HALF_SCAN_TIME = 0.069000
    SYNC_PULSE = 0.009000
    SYNC_PORCH = 0.003000
    SEP_PULSE = 0.004500
    SEP_PORCH = 0.001500
    CHAN_COUNT = 3
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    HALF_CHAN_TIME = SEP_PULSE + HALF_SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME + SEP_PORCH)
    CHAN_OFFSETS.append(CHAN_OFFSETS[1] + HALF_CHAN_TIME + SEP_PORCH)
    LINE_TIME = CHAN_OFFSETS[2] + HALF_SCAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    HALF_PIXEL_TIME = HALF_SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 4.88
    HAS_ALT_SCAN = False

class PD50(object):
    """PD50 SSTV mode

    Parameters chosen to produce approximately 50s image duration for 256 lines
    using 3 color channels (R,G,B). Values follow typical SSTV timing conventions
    (9ms sync, 1.5ms porch, 1.5ms separator) and compute channel/line timings
    so LINE_TIME * LINE_COUNT ~= 50s.
    """
    NAME = "PD50"
    COLOR = COL_FMT.RGB
    LINE_WIDTH = 320
    LINE_COUNT = 256
    SYNC_PULSE = 0.009000
    SYNC_PORCH = 0.001500
    SEP_PULSE = 0.001500
    CHAN_COUNT = 3
    CHAN_SYNC = 0
    # SCAN_TIME computed so total image ~50s: LINE_TIME = SYNC_PULSE + SYNC_PORCH + 3*(SEP_PULSE+SCAN_TIME)
    SCAN_TIME = 0.06060417
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME)
    CHAN_OFFSETS.append(CHAN_OFFSETS[1] + CHAN_TIME)
    LINE_TIME = SYNC_PULSE + SYNC_PORCH + 3 * CHAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 3.0
    HAS_START_SYNC = True
    HAS_HALF_SCAN = False
    HAS_ALT_SCAN = False

class PD90(object):
    """PD90 SSTV mode

    Parameters chosen to produce approximately 90s image duration for 256 lines
    using 3 color channels (R,G,B). Timing uses same sync/porch/separator values
    and larger SCAN_TIME so LINE_TIME * LINE_COUNT ~= 90s.
    """
    NAME = "PD90"
    COLOR = COL_FMT.RGB
    LINE_WIDTH = 320
    LINE_COUNT = 256
    SYNC_PULSE = 0.009000
    SYNC_PORCH = 0.001500
    SEP_PULSE = 0.001500
    CHAN_COUNT = 3
    CHAN_SYNC = 0
    SCAN_TIME = 0.11268750
    CHAN_TIME = SEP_PULSE + SCAN_TIME
    CHAN_OFFSETS = [SYNC_PULSE + SYNC_PORCH]
    CHAN_OFFSETS.append(CHAN_OFFSETS[0] + CHAN_TIME)
    CHAN_OFFSETS.append(CHAN_OFFSETS[1] + CHAN_TIME)
    LINE_TIME = SYNC_PULSE + SYNC_PORCH + 3 * CHAN_TIME
    PIXEL_TIME = SCAN_TIME / LINE_WIDTH
    WINDOW_FACTOR = 3.0
    HAS_START_SYNC = True
    HAS_HALF_SCAN = False
    HAS_ALT_SCAN = False

VIS_MAP = {8: R36,
           12: R72,
           40: M2,
           44: M1,
           56: S2,
           60: S1,
           76: SDX,
           93: PD50,
           108: PD90}

BREAK_OFFSET = 0.300
LEADER_OFFSET = 0.010 + BREAK_OFFSET
VIS_START_OFFSET = 0.300 + LEADER_OFFSET
HDR_SIZE = 0.030 + VIS_START_OFFSET
HDR_WINDOW_SIZE = 0.010
VIS_BIT_SIZE = 0.030

"""Parsing arguments and starting program from command line"""

class SSTVCommand(object):
    """Main class to handle the command line features"""
    examples_of_use = """
examples:
  Decode local SSTV audio file named 'audio.wav' to 'result.png':
    $ sstv -d audio.wav

  Decode SSTV audio file in /tmp to './image.jpg':
    $ sstv -d /tmp/signal.wav -o ./image.jpg

  Start decoding SSTV signal at 50.5 seconds into the audio
    $ sstv -d audio.wav -s 50.50"""

    def __init__(self, shell_args=None):
        """Handle command line arguments"""

        self._audio_file = None
        self._output_file = None

        if shell_args is None:
            self.args = self.parse_args(argv[1:])
        else:
            self.args = self.parse_args(shell_args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def __del__(self):
        self.close()

    def init_args(self):
        """Initialise argparse parser"""
        
        parser = argparse.ArgumentParser(
            prog="sstv",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.examples_of_use)

        parser.add_argument("-d", "--decode", type=argparse.FileType('rb'),
                            help="decode SSTV audio file", dest="audio_file")
        parser.add_argument("-o", "--output", type=str,
                            help="save output image to custom filename",
                            default="result.png", dest="output_file")
        parser.add_argument("--dir", type=str, default="", dest="output_dir",
                            help=f"directory for save decoded image (default: same directory with app)")
        parser.add_argument("-s", "--skip", type=float,
                            help="time in seconds to start decoding signal at",
                            default=0.0, dest="skip")
        parser.add_argument("-V", "--version", action="version",
                            version=f"sstv2satno-v{VERSION}\nforked from colaclanth <https://github.com/colaclanth/sstv>\nby hobisatelit <https://github.com/hobisatelit/sstv2satno>")
        parser.add_argument("--list-modes", action="store_true",
                            dest="list_modes",
                            help="list supported SSTV modes")
        parser.add_argument("--list-audio-formats", action="store_true",
                            dest="list_audio_formats",
                            help="list supported audio file formats")
        parser.add_argument("--list-image-formats", action="store_true",
                            dest="list_image_formats",
                            help="list supported image file formats")
        parser.add_argument("--slant", type=float, default="0.0", dest="slant",
                            help=f"custom slant factor, example for SONATE2 it should be -0.45. Override auto if non-zero")
        return parser

    def parse_args(self, shell_args):
        global custom_slant
        """Parse command line arguments"""

        parser = self.init_args()
        args = parser.parse_args(shell_args)

        self._audio_file = args.audio_file
        self._output_file = args.output_file
        self._output_dir = args.output_dir
        self._skip = args.skip
        self._slant = args.slant

        if args.list_modes:
            self.list_supported_modes()
            exit(0)
        if args.list_audio_formats:
            self.list_supported_audio_formats()
            exit(0)
        if args.list_image_formats:
            self.list_supported_image_formats()
            exit(0)

        if self._audio_file is None:
            parser.print_help()
            exit(2)

        return args

    def start(self):
        """Start decoder"""

        with SSTVDecoder(self._audio_file, slant=self._slant) as sstv:
            images = sstv.decode(self._skip)
            if not images:
                exit(2)
                
            if sys.platform == 'win32':
                script_dir = os.getcwd()
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
            output_dir = os.path.join(script_dir, self._output_dir)
            os.makedirs(output_dir, exist_ok=True)
            log_message(f"Output_dir: {output_dir}")
            for idx, (img, mode_name) in enumerate(images, 1):
                formatted_idx = f"{idx:03d}"
                safe_mode = mode_name.replace(" ", "").replace("-","" ).lower()
                base_name = self._output_file.rsplit('.', 1)

                if len(base_name) == 2:
                    output_filename = f"{base_name[0]}_{formatted_idx}_{safe_mode}.{base_name[1]}"
                else:
                    output_filename = f"output_{formatted_idx}_{safe_mode}.png"

                try:
                    # Auto slant correction
                    corrected_img = sstv._auto_correct_slant(img)
                    # ENHANCED: Apply brightness and contrast before saving
                    enhanced_img = self._enhance_image(corrected_img)
                    #enhanced_img = self._enhance_image(img)
                    enhanced_img.save(os.path.join(output_dir, output_filename))
                except (KeyError, ValueError) as e:
                    log_message(f"Error saving Image {idx}, saved to output-{formatted_idx}-{safe_mode}.png instead",
                                err=True)
                    enhanced_img = self._enhance_image(img)
                    enhanced_img.save(os.path.join(output_dir, f"output_{formatted_idx}-{safe_mode}.png"))

],