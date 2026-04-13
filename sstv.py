#!/usr/bin/env python3
# forked from: https://github.com/colaclanth/sstv
# Copyright 2026 colaclanth, hobisatelit
# https://github.com/hobisatelit/sstv2satno
# License: GPL-3.0-or-later

VERSION = "0.1"

import sys
import os
import signal
import wave
from enum import Enum
import argparse
import numpy as np
from PIL import Image
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
    SCAN_TIME = 0.146432
    SYNC_PULSE = 0.004862
    SYNC_PORCH = 0.000572
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
    WINDOW_FACTOR = 7.70
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

VIS_MAP = {8: R36,
           12: R72,
           40: M2,
           44: M1,
           56: S2,
           60: S1,
           76: SDX}

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
                            help="save output image to custom location",
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
        return parser

    def parse_args(self, shell_args):
        """Parse command line arguments"""

        parser = self.init_args()
        args = parser.parse_args(shell_args)

        self._audio_file = args.audio_file
        self._output_file = args.output_file
        self._output_dir = args.output_dir
        self._skip = args.skip

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

        with SSTVDecoder(self._audio_file) as sstv:
            images = sstv.decode(self._skip)
            if not images:  # No SSTV signal found
                exit(2)
                
            if sys.platform == 'win32':
                script_dir = os.getcwd()
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
            output_dir = os.path.join(script_dir, self._output_dir)
            os.makedirs(output_dir, exist_ok=True)
            print(f"output_dir: {output_dir}")
            for idx, img in enumerate(images, 1):
                formatted_idx = f"{idx:03d}"
                # Generate filename with prefix and counter
                base_name = self._output_file.rsplit('.', 1)

                if len(base_name) == 2:
                    output_filename = f"{base_name[0]}_{formatted_idx}.{base_name[1]}"
                else:
                    output_filename = f"output_{formatted_idx}.png"

                try:
                    img.save(os.path.join(output_dir,output_filename))
                    log_message(f"Image {formatted_idx} saved as {os.path.join(output_dir,output_filename)}")
                except (KeyError, ValueError):
                    log_message(f"Error saving file {formatted_idx}, saved to output-{formatted_idx}.png instead",
                                err=True)
                    img.save(os.path.join(output_dir,f"output_{formatted_idx}.png"))

    def close(self):
        """Closes any input/output files if they exist"""

        if self._audio_file is not None and not self._audio_file.closed:
            self._audio_file.close()

    def list_supported_modes(self):
        modes = ', '.join([fmt.NAME for fmt in VIS_MAP.values()])
        print("Supported modes: {}".format(modes))

    def list_supported_audio_formats(self):
        print("Supported audio formats: wav")

    def list_supported_image_formats(self):
        Image.init()
        image_formats = ', '.join(Image.SAVE.keys())
        print("Supported image formats: {}".format(image_formats))
        
       
"""Class and methods to decode SSTV signal"""

def calc_lum(freq):
    """Converts SSTV pixel frequency range into 0-255 luminance byte"""

    lum = int(round((freq - 1500) / 3.1372549))
    return min(max(lum, 0), 255)


def barycentric_peak_interp(bins, x):
    """Interpolate between frequency bins to find x value of peak"""

    # Takes x as the index of the largest bin and interpolates the
    # x value of the peak using neighbours in the bins array

    # Make sure data is in bounds
    y1 = bins[x] if x <= 0 else bins[x-1]
    y3 = bins[x] if x + 1 >= len(bins) else bins[x+1]

    denom = y3 + bins[x] + y1
    if denom == 0:
        return 0  # erroneous

    return (y3 - y1) / denom + x


class SSTVDecoder(object):

    """Create an SSTV decoder for decoding audio data"""

    def __init__(self, audio_file):
        self.mode = None
        self._audio_file = audio_file

        # Read WAV file using built-in wave module
        try:
            with wave.open(self._audio_file, 'rb') as wav_file:
                self._sample_rate = wav_file.getframerate()
                n_channels = wav_file.getnchannels()
                n_frames = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                audio_data = wav_file.readframes(n_frames)
            
            # Convert bytes to numpy array
            dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
            dtype = dtype_map.get(sample_width, np.int16)
            self._samples = np.frombuffer(audio_data, dtype=dtype).astype(float)
            
            # Normalize to [-1, 1] range
            if dtype == np.uint8:
                self._samples = (self._samples - 128) / 128.0
            else:
                self._samples = self._samples / (2 ** (sample_width * 8 - 1))
            
            # Convert to mono if stereo
            if n_channels > 1:
                self._samples = self._samples.reshape(-1, n_channels).mean(axis=1)
        
        except wave.Error as e:
            raise ValueError(f"Error reading WAV file: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def __del__(self):
        self.close()

    def decode(self, skip=0.0):
        """Attempts to decode the audio data as SSTV signals
        
        Returns a list of PIL images on success, and empty list if no SSTV signals were found
        """

        if skip > 0.0:
            self._samples = self._samples[round(skip * self._sample_rate):]

        decoded_images = []
        search_start = 0
        image_count = 0

        while True:
            # Search for header starting from current position
            header_end = self._find_header_from(search_start)

            if header_end is None:
                break  # No more headers found

            try:
                self.mode = self._decode_vis(header_end)

                vis_end = header_end + round(VIS_BIT_SIZE * 9 * self._sample_rate)

                image_data = self._decode_image_data(vis_end)

                image = self._draw_image(image_data)
                decoded_images.append(image)
                image_count += 1

                log_message("\nImage {} decoded successfully!".format(image_count))

                # Set search start to end of current image data plus some margin
                search_start = vis_end + round(self.mode.LINE_TIME * 
                                              self.mode.LINE_COUNT * self._sample_rate) + 1000

            except (ValueError, EOFError) as e:
                log_message("Error decoding image: {}".format(str(e)), err=True)
                # Skip ahead and continue searching
                search_start = header_end + 1000
                continue

        if not decoded_images:
            log_message("Couldn't find any SSTV images in the given audio file", err=True)
        else:
            log_message("\nTotal images decoded: {}".format(image_count))

        return decoded_images

    def close(self):
        """Closes any input files if they exist"""

        if self._audio_file is not None and not self._audio_file.closed:
            self._audio_file.close()

    def _peak_fft_freq(self, data):
        """Finds the peak frequency from a section of audio data"""

        windowed_data = data * hann(len(data))
        fft = np.abs(np.fft.rfft(windowed_data))

        # Get index of bin with highest magnitude
        x = np.argmax(fft)
        # Interpolated peak frequency
        peak = barycentric_peak_interp(fft, x)

        # Return frequency in hz
        return peak * self._sample_rate / len(windowed_data)

    def _find_header(self):
        """Finds the approx sample of the end of the calibration header"""

        return self._find_header_from(0)

    def _find_header_from(self, search_start):
        """Finds the approx sample of the end of the calibration header starting from search_start"""

        header_size = round(HDR_SIZE * self._sample_rate)
        window_size = round(HDR_WINDOW_SIZE * self._sample_rate)

        # Relative sample offsets of the header tones
        leader_1_sample = 0
        leader_1_search = leader_1_sample + window_size

        break_sample = round(BREAK_OFFSET * self._sample_rate)
        break_search = break_sample + window_size

        leader_2_sample = round(LEADER_OFFSET * self._sample_rate)
        leader_2_search = leader_2_sample + window_size

        vis_start_sample = round(VIS_START_OFFSET * self._sample_rate)
        vis_start_search = vis_start_sample + window_size

        jump_size = round(0.002 * self._sample_rate)  # check every 2ms

        # The margin of error created here will be negligible when decoding the
        # vis due to each bit having a length of 30ms. We fix this error margin
        # when decoding the image by aligning each sync pulse

        for current_sample in range(search_start, len(self._samples) - header_size,
                                    jump_size):
            # Update search progress message
            if current_sample % (jump_size * 256) == 0:
                search_msg = "Searching for calibration header... {:.1f}s"
                progress = current_sample / self._sample_rate
                log_message(search_msg.format(progress), recur=True)

            search_end = current_sample + header_size
            search_area = self._samples[current_sample:search_end]

            leader_1_area = search_area[leader_1_sample:leader_1_search]
            break_area = search_area[break_sample:break_search]
            leader_2_area = search_area[leader_2_sample:leader_2_search]
            vis_start_area = search_area[vis_start_sample:vis_start_search]

            # Check they're the correct frequencies
            if (abs(self._peak_fft_freq(leader_1_area) - 1900) < 50
               and abs(self._peak_fft_freq(break_area) - 1200) < 50
               and abs(self._peak_fft_freq(leader_2_area) - 1900) < 50
               and abs(self._peak_fft_freq(vis_start_area) - 1200) < 50):

                stop_msg = "Searching for calibration header... Found!{:>4}"
                log_message(stop_msg.format(' '))
                return current_sample + header_size

        log_message()
        return None

    def _decode_vis(self, vis_start):
        """Decodes the vis from the audio data and returns the SSTV mode"""

        bit_size = round(VIS_BIT_SIZE * self._sample_rate)
        vis_bits = []

        for bit_idx in range(8):
            bit_offset = vis_start + bit_idx * bit_size
            section = self._samples[bit_offset:bit_offset+bit_size]
            freq = self._peak_fft_freq(section)
            # 1100 hz = 1, 1300hz = 0
            vis_bits.append(int(freq <= 1200))

        # Check for even parity in last bit
        parity = sum(vis_bits) % 2 == 0
        if not parity:
            raise ValueError("Error decoding VIS header (invalid parity bit)")

        # LSB first so we must reverse and ignore the parity bit
        vis_value = 0
        for bit in vis_bits[-2::-1]:
            vis_value = (vis_value << 1) | bit

        if vis_value not in VIS_MAP:
            error = "SSTV mode is unsupported (VIS: {})"
            raise ValueError(error.format(vis_value))

        mode = VIS_MAP[vis_value]
        log_message("Detected SSTV mode {}".format(mode.NAME))

        return mode

    def _align_sync(self, align_start, start_of_sync=True):
        """Returns sample where the beginning of the sync pulse was found"""

        # TODO - improve this

        sync_window = round(self.mode.SYNC_PULSE * 1.4 * self._sample_rate)
        align_stop = len(self._samples) - sync_window

        if align_stop <= align_start:
            return None  # Reached end of audio

        for current_sample in range(align_start, align_stop):
            section_end = current_sample + sync_window
            search_section = self._samples[current_sample:section_end]

            if self._peak_fft_freq(search_section) > 1350:
                break

        end_sync = current_sample + (sync_window // 2)

        if start_of_sync:
            return end_sync - round(self.mode.SYNC_PULSE * self._sample_rate)
        else:
            return end_sync

    def _decode_image_data(self, image_start):
        """Decodes image from the transmission section of an sstv signal"""

        window_factor = self.mode.WINDOW_FACTOR
        centre_window_time = (self.mode.PIXEL_TIME * window_factor) / 2
        pixel_window = round(centre_window_time * 2 * self._sample_rate)

        height = self.mode.LINE_COUNT
        channels = self.mode.CHAN_COUNT
        width = self.mode.LINE_WIDTH
        # Use list comprehension to init list so we can return data early
        image_data = [[[0 for i in range(width)]
                       for j in range(channels)] for k in range(height)]

        seq_start = image_start
        if self.mode.HAS_START_SYNC:
            # Start at the end of the initial sync pulse
            seq_start = self._align_sync(image_start, start_of_sync=False)
            if seq_start is None:
                raise EOFError("Reached end of audio before image data")

        for line in range(height):

            if self.mode.CHAN_SYNC > 0 and line == 0:
                # Align seq_start to the beginning of the previous sync pulse
                sync_offset = self.mode.CHAN_OFFSETS[self.mode.CHAN_SYNC]
                seq_start -= round((sync_offset + self.mode.SCAN_TIME)
                                   * self._sample_rate)

            for chan in range(channels):

                if chan == self.mode.CHAN_SYNC:
                    if line > 0 or chan > 0:
                        # Set base offset to the next line
                        seq_start += round(self.mode.LINE_TIME *
                                           self._sample_rate)

                    # Align to start of sync pulse
                    seq_start = self._align_sync(seq_start)
                    if seq_start is None:
                        log_message()
                        log_message("Reached end of audio whilst decoding.")
                        return image_data

                pixel_time = self.mode.PIXEL_TIME
                if self.mode.HAS_HALF_SCAN:
                    # Robot mode has half-length second/third scans
                    if chan > 0:
                        pixel_time = self.mode.HALF_PIXEL_TIME

                    centre_window_time = (pixel_time * window_factor) / 2
                    pixel_window = round(centre_window_time * 2 *
                                         self._sample_rate)

                for px in range(width):

                    chan_offset = self.mode.CHAN_OFFSETS[chan]

                    px_pos = round(seq_start + (chan_offset + px *
                                   pixel_time - centre_window_time) *
                                   self._sample_rate)
                    px_end = px_pos + pixel_window

                    # If we are performing fft past audio length, stop early
                    if px_end >= len(self._samples):
                        log_message()
                        log_message("Reached end of audio whilst decoding.")
                        return image_data

                    pixel_area = self._samples[px_pos:px_end]
                    freq = self._peak_fft_freq(pixel_area)

                    image_data[line][chan][px] = calc_lum(freq)

            progress_bar(line, height - 1, "Decoding image...")

        return image_data

    def _draw_image(self, image_data):
        """Renders the image from the decoded sstv signal"""

        # Let PIL do YUV-RGB conversion for us
        if self.mode.COLOR == COL_FMT.YUV:
            col_mode = "YCbCr"
        else:
            col_mode = "RGB"

        width = self.mode.LINE_WIDTH
        height = self.mode.LINE_COUNT
        channels = self.mode.CHAN_COUNT

        image = Image.new(col_mode, (width, height))
        pixel_data = image.load()

        log_message("Drawing image data...")

        for y in range(height):

            odd_line = y % 2
            for x in range(width):

                if channels == 2:

                    if self.mode.HAS_ALT_SCAN:
                        if self.mode.COLOR == COL_FMT.YUV:
                            # R36
                            pixel = (image_data[y][0][x],
                                     image_data[y-(odd_line-1)][1][x],
                                     image_data[y-odd_line][1][x])

                elif channels == 3:

                    if self.mode.COLOR == COL_FMT.GBR:
                        # M1, M2, S1, S2, SDX
                        pixel = (image_data[y][2][x],
                                 image_data[y][0][x],
                                 image_data[y][1][x])
                    elif self.mode.COLOR == COL_FMT.YUV:
                        # R72
                        pixel = (image_data[y][0][x],
                                 image_data[y][2][x],
                                 image_data[y][1][x])
                    elif self.mode.COLOR == COL_FMT.RGB:
                        pixel = (image_data[y][0][x],
                                 image_data[y][1][x],
                                 image_data[y][2][x])

                pixel_data[x, y] = pixel

        if image.mode != "RGB":
            image = image.convert("RGB")

        log_message("...Done!")
        return image


"""Shared methods"""

def log_message(message="", show=True, err=False, recur=False, prefix=True):
    """Simple print wrapper"""

    if not show:
        return
    out = stdout
    if err:
        out = stderr
    end = '\n'
    if recur:
        end = '\r'
        if platform == "win32":
            message = ''.join(['\r', message])
        cols = get_terminal_size().columns
        if cols < len(message):
            message = message[:cols]
    if prefix:
        message = ' '.join(["[sstv]", message])

    print(message, file=out, end=end)


def progress_bar(progress, complete, message="", show=True):
    """Simple loading bar"""

    if not show:
        return

    message_size = len(message) + 7  # prefix size
    cols = get_terminal_size().columns
    percent_on = True
    level = progress / complete
    bar_size = min(cols - message_size - 10, 100)
    bar = ""

    if bar_size > 5:
        fill_size = round(bar_size * level)
        bar = "[{}]".format(''.join(['#' * fill_size,
                                     '.' * (bar_size - fill_size)]))
    elif bar_size < -3:
        percent_on = False

    percent = ""
    if percent_on:
        percent = "{:4d}%".format(int(level * 100))

    align = cols - message_size - len(percent)
    not_end = not progress == complete
    log_message("{}{:>{width}}{}".format(message, bar, percent, width=align),
                recur=not_end)

def handle_sigint(sig, frame):
    print()
    log_message("Received interrupt signal, exiting.")
    exit(0)

def main():
    signal.signal(signal.SIGINT, handle_sigint)
    with SSTVCommand() as prog:
        prog.start()

if __name__ == "__main__":
    main()
