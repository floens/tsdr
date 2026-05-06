"""PAD (Programme Associated Data) decoder for DAB+.

Extracts DLS (Dynamic Label Segment) text and MOT (Multimedia Object Transfer)
slideshow images from X-PAD/F-PAD carried in AAC access units.

Reference: ETSI EN 300 401 §7.4, ETSI TS 101 756, ETSI EN 301 234.
Reference: ETSI EN 301 234 (MOT).
"""

from .charset import decode_charset as _decode_charset
from .fec import _crc16_bytes

CRC_LEN = 2
_XpadCi_LENS = (4, 6, 8, 12, 16, 24, 32, 48)


# Data group base


class _DataGroup:
    """Accumulates data subfields across multiple X-PAD frames into a complete
    data group, then validates CRC and calls _decode().
    """

    def __init__(self, max_size: int) -> None:
        self._buf = bytearray(max_size)
        self._size = 0
        self._needed = self._initial_needed()

    def _initial_needed(self) -> int:
        return 0

    def _decode(self) -> bool:
        raise NotImplementedError

    def reset(self) -> None:
        self._size = 0
        self._needed = self._initial_needed()

    def process_subfield(self, start: bool, data: bytes) -> bool:
        """Append a data subfield. Returns True when a complete data group is decoded."""
        if start:
            self.reset()
        elif self._size == 0:
            return False

        if self._size >= self._needed:
            return False
        if self._size >= len(self._buf):
            return False

        copy_len = min(len(data), len(self._buf) - self._size)
        self._buf[self._size : self._size + copy_len] = data[:copy_len]
        self._size += copy_len

        if self._size < self._needed:
            return False
        return self._decode()

    def _ensure_size(self, desired: int) -> bool:
        self._needed = desired
        return self._size >= self._needed

    def _check_crc(self, data_len: int) -> bool:
        if self._size < data_len + CRC_LEN:
            return False
        crc_stored = self._buf[data_len] << 8 | self._buf[data_len + 1]
        # CRC-16 CCITT with ones' complement (final XOR 0xFFFF)
        crc_calced = _crc16_bytes(bytes(self._buf[:data_len])) ^ 0xFFFF
        return crc_stored == crc_calced


# DGLI (Data Group Length Indicator)


class _DGLIDecoder(_DataGroup):
    def __init__(self) -> None:
        super().__init__(2 + CRC_LEN)
        self.dgli_len = 0

    def _initial_needed(self) -> int:
        return 2 + CRC_LEN

    def _decode(self) -> bool:
        if not self._check_crc(2):
            self.reset()
            return False
        self.dgli_len = (self._buf[0] & 0x3F) << 8 | self._buf[1]
        self.reset()
        return True

    def get_len(self) -> int:
        result = self.dgli_len
        self.dgli_len = 0
        return result


# DLS (Dynamic Label Segment)


class _DLSegment:
    __slots__ = ("prefix", "chars")

    def __init__(self, prefix: bytes, chars: bytes) -> None:
        self.prefix = prefix
        self.chars = chars

    @property
    def toggle(self) -> bool:
        return bool(self.prefix[0] & 0x80)

    @property
    def first(self) -> bool:
        return bool(self.prefix[0] & 0x40)

    @property
    def last(self) -> bool:
        return bool(self.prefix[0] & 0x20)

    @property
    def seg_num(self) -> int:
        return 0 if self.first else ((self.prefix[1] & 0x70) >> 4)


class _DLReassembler:
    def __init__(self) -> None:
        self.segments: dict[int, _DLSegment] = {}
        self.label_raw = b""
        self.charset = -1

    def reset(self) -> None:
        self.segments.clear()
        self.label_raw = b""
        self.charset = -1

    def add_segment(self, seg: _DLSegment) -> bool:
        """Add a segment. Returns True if the label is now complete."""
        # Clear cache if toggle changed
        if self.segments:
            existing = next(iter(self.segments.values()))
            if existing.toggle != seg.toggle:
                self.segments.clear()

        if seg.seg_num in self.segments:
            return False

        self.segments[seg.seg_num] = seg
        return self._check_complete()

    def _check_complete(self) -> bool:
        parts = []
        for i in range(8):
            seg = self.segments.get(i)
            if seg is None:
                return False
            parts.append(seg.chars)
            if seg.last:
                self.label_raw = b"".join(parts)
                self.charset = self.segments[0].prefix[1] >> 4
                return True
        return False


class _DynamicLabelDecoder(_DataGroup):
    def __init__(self) -> None:
        super().__init__(2 + 16 + CRC_LEN)  # prefix + max 16 chars + CRC
        self._reassembler = _DLReassembler()
        self.label: str | None = None
        self.charset: int = -1

    def _initial_needed(self) -> int:
        return 2 + CRC_LEN  # at least prefix + CRC

    def reset_all(self) -> None:
        self.reset()
        self._reassembler.reset()
        self.label = None
        self.charset = -1

    def _decode(self) -> bool:
        command = bool(self._buf[0] & 0x10)

        if command:
            if (self._buf[0] & 0x0F) == 0x01:
                # Remove label command
                self.label = ""
                self.charset = -1
                return True
            self.reset()
            return False

        field_len = (self._buf[0] & 0x0F) + 1
        real_len = 2 + field_len

        if not self._ensure_size(real_len + CRC_LEN):
            return False

        if not self._check_crc(real_len):
            self.reset()
            return False

        seg = _DLSegment(
            prefix=bytes(self._buf[:2]),
            chars=bytes(self._buf[2 : 2 + field_len]),
        )
        self.reset()

        if not self._reassembler.add_segment(seg):
            return False

        raw = self._reassembler.label_raw
        self.charset = self._reassembler.charset
        self.label = _decode_charset(raw, self.charset)
        return True


# MOT (Multimedia Object Transfer)


class _MOTDataGroupDecoder(_DataGroup):
    """Accumulates MOT data group bytes (length set by DGLI)."""

    def __init__(self) -> None:
        self.mot_len = 0
        self._result: bytes = b""
        super().__init__(16384)  # 2^14 max

    def _initial_needed(self) -> int:
        return self.mot_len

    def set_len(self, mot_len: int) -> None:
        self.mot_len = mot_len
        self._needed = mot_len

    def _decode(self) -> bool:
        if self.mot_len < CRC_LEN:
            return False
        if not self._check_crc(self.mot_len - CRC_LEN):
            self.reset()
            return False
        self._result = bytes(self._buf[: self.mot_len])
        self.reset()
        return True

    def get_data_group(self) -> bytes:
        return self._result


class _MOTEntity:
    """Collects segments (header or body) for one MOT object."""

    def __init__(self) -> None:
        self.segments: dict[int, bytes] = {}
        self.last_seg_num: int = -1

    def add_segment(self, seg_num: int, data: bytes, last: bool) -> None:
        self.segments[seg_num] = data
        if last:
            self.last_seg_num = seg_num

    def is_finished(self) -> bool:
        if self.last_seg_num < 0:
            return False
        return all(i in self.segments for i in range(self.last_seg_num + 1))

    def get_data(self) -> bytes:
        parts = []
        for i in range(self.last_seg_num + 1):
            parts.append(self.segments[i])
        return b"".join(parts)

    def reset(self) -> None:
        self.segments.clear()
        self.last_seg_num = -1


# Content type constants
_CONTENT_TYPE_IMAGE = 0x02
_CONTENT_SUB_TYPE_JFIF = 0x001
_CONTENT_SUB_TYPE_PNG = 0x003


class MOTFile:
    """A completed MOT object (slideshow image)."""

    __slots__ = (
        "data",
        "body_size",
        "content_type",
        "content_sub_type",
        "content_name",
        "category_title",
        "click_through_url",
        "trigger_time_now",
    )

    def __init__(self) -> None:
        self.data: bytes = b""
        self.body_size: int = 0
        self.content_type: int = 0
        self.content_sub_type: int = 0
        self.content_name: str = ""
        self.category_title: str = ""
        self.click_through_url: str = ""
        self.trigger_time_now: bool = True


class _MOTObject:
    """Single MOT object being assembled (header + body)."""

    def __init__(self) -> None:
        self.header = _MOTEntity()
        self.body = _MOTEntity()

    def reset(self) -> None:
        self.header.reset()
        self.body.reset()


class _MOTManager:
    """Manages assembly of MOT objects from data groups."""

    def __init__(self) -> None:
        self._objects: dict[int, _MOTObject] = {}
        self._file: MOTFile | None = None

    def reset(self) -> None:
        self._objects.clear()
        self._file = None

    def handle_data_group(self, dg: bytes) -> bool:
        """Process a MOT data group. Returns True if a new file is complete."""
        if len(dg) < 2:
            return False

        # Data group header
        header_byte = dg[0]
        crc_flag = bool(header_byte & 0x40)
        segment_flag = bool(header_byte & 0x20)
        user_access_flag = bool(header_byte & 0x10)
        dg_type = header_byte & 0x0F

        if not (crc_flag and segment_flag and user_access_flag):
            return False
        if dg_type not in (3, 4):  # 3=header, 4=body
            return False

        pos = 1  # skip header extension byte

        # Segment header
        if pos >= len(dg):
            return False
        # Skip header extension (1 byte)
        pos = 2
        if pos + 2 > len(dg):
            return False

        seg_info = (dg[pos] << 8) | dg[pos + 1]
        last_seg = bool(seg_info & 0x8000)
        seg_num = seg_info & 0x7FFF
        pos += 2

        # User access header
        if pos >= len(dg):
            return False
        transport_id_flag = bool(dg[pos] & 0x10)
        len_indicator = dg[pos] & 0x0F
        pos += 1

        if not transport_id_flag or len_indicator < 2:
            return False
        if pos + len_indicator > len(dg):
            return False

        transport_id = (dg[pos] << 8) | dg[pos + 1]
        pos += len_indicator

        # Segmentation header (2 bytes: 3 reserved bits + 13-bit segment size)
        if pos + 2 > len(dg):
            return False
        seg_size = ((dg[pos] & 0x1F) << 8) | dg[pos + 1]
        pos += 2

        # Segment data
        seg_data = dg[pos : pos + seg_size]

        # Get or create object
        obj = self._objects.get(transport_id)
        if obj is None:
            obj = _MOTObject()
            self._objects[transport_id] = obj

        if dg_type == 3:
            obj.header.add_segment(seg_num, seg_data, last_seg)
        else:
            obj.body.add_segment(seg_num, seg_data, last_seg)

        if not (obj.header.is_finished() and obj.body.is_finished()):
            return False

        # Parse completed object
        header_data = obj.header.get_data()
        body_data = obj.body.get_data()

        mot_file = self._parse_header(header_data)
        if mot_file is None:
            del self._objects[transport_id]
            return False

        mot_file.data = body_data
        if mot_file.body_size != 0 and len(body_data) != mot_file.body_size:
            del self._objects[transport_id]
            return False

        if not mot_file.trigger_time_now:
            return False

        self._file = mot_file
        del self._objects[transport_id]
        return True

    def get_file(self) -> MOTFile | None:
        return self._file

    def _parse_header(self, data: bytes) -> MOTFile | None:
        if len(data) < 7:
            return None

        f = MOTFile()
        # Body size: 28 bits
        f.body_size = (data[0] << 20) | (data[1] << 12) | (data[2] << 4) | (data[3] >> 4)
        # Header size: 13 bits
        header_size = ((data[3] & 0x0F) << 9) | (data[4] << 1) | (data[5] >> 7)
        # Content type: 6 bits
        f.content_type = (data[5] & 0x7E) >> 1
        # Content sub-type: 9 bits
        f.content_sub_type = ((data[5] & 0x01) << 8) | data[6]

        # Parse header extensions
        pos = 7
        end = min(7 + header_size, len(data))
        while pos < end:
            if pos >= len(data):
                break
            pli = (data[pos] >> 6) & 0x03
            param_id = data[pos] & 0x3F
            pos += 1

            if pli == 0b00:
                param_data = b""
            elif pli == 0b01:
                if pos >= end:
                    break
                param_data = data[pos : pos + 1]
                pos += 1
            elif pli == 0b10:
                if pos + 4 > end:
                    break
                param_data = data[pos : pos + 4]
                pos += 4
            else:  # 0b11 variable
                if pos >= end:
                    break
                ext_len = data[pos]
                pos += 1
                if pos + ext_len > end:
                    break
                param_data = data[pos : pos + ext_len]
                pos += ext_len

            if param_id == 0x05 and len(param_data) >= 1:
                # TriggerTime: bit 7=0 means NOW
                f.trigger_time_now = not bool(param_data[0] & 0x80)
            elif param_id == 0x0C and len(param_data) >= 2:
                # ContentName: charset (4 bits) + name
                cn_charset = param_data[0] >> 4
                f.content_name = _decode_charset(param_data[1:], cn_charset)
            elif param_id == 0x26 and len(param_data) >= 1:
                # CategoryTitle (UTF-8)
                f.category_title = param_data.decode("utf-8", errors="replace")
            elif param_id == 0x27 and len(param_data) >= 1:
                # ClickThroughURL (UTF-8)
                f.click_through_url = param_data.decode("utf-8", errors="replace")

        return f


# XPAD_CI (Content Indicator)


class _XpadCi:
    __slots__ = ("len", "type")

    def __init__(self, length: int = 0, ci_type: int = -1) -> None:
        self.len = length
        self.type = ci_type

    @classmethod
    def from_raw(cls, raw: int) -> _XpadCi:
        return cls(length=_XpadCi_LENS[raw >> 5], ci_type=raw & 0x1F)


# PAD Decoder (top-level)


class PADDecoder:
    """Decodes PAD (F-PAD + X-PAD) from DAB+ audio frames.

    Call process() once per AU with extracted PAD data.
    Check dynamic_label and slide properties for new data.
    """

    def __init__(self) -> None:
        self._mot_app_type: int = -1
        self._last_xpad_ci = _XpadCi()
        self._dl_decoder = _DynamicLabelDecoder()
        self._dgli_decoder = _DGLIDecoder()
        self._mot_decoder = _MOTDataGroupDecoder()
        self._mot_manager = _MOTManager()

        self.dynamic_label: str | None = None
        self.label_changed: bool = False
        self.slide: MOTFile | None = None
        self.slide_changed: bool = False

    @property
    def mot_app_type(self) -> int:
        return self._mot_app_type

    @mot_app_type.setter
    def mot_app_type(self, value: int) -> None:
        self._mot_app_type = value

    def reset(self) -> None:
        self._mot_app_type = -1
        self._last_xpad_ci = _XpadCi()
        self._dl_decoder.reset_all()
        self._dgli_decoder.reset()
        self._mot_decoder.reset()
        self._mot_manager.reset()
        self.dynamic_label = None
        self.label_changed = False
        self.slide = None
        self.slide_changed = False

    def process(self, xpad_data: bytes, fpad_data: bytes) -> None:
        """Process PAD from one AU. X-PAD is in reversed byte order (as from DSE)."""
        self.label_changed = False
        self.slide_changed = False

        # Reverse X-PAD byte order
        xpad_len = len(xpad_data)
        xpad = bytes(reversed(xpad_data[: min(xpad_len, 196)]))

        # Parse F-PAD
        fpad_type = fpad_data[0] >> 6
        xpad_ind = (fpad_data[0] & 0x30) >> 4
        ci_flag = bool(fpad_data[1] & 0x02)

        prev_ci = self._last_xpad_ci
        self._last_xpad_ci = _XpadCi()

        # Build CI list
        cis: list[_XpadCi] = []
        cis_len = 0

        if fpad_type == 0b00:
            if ci_flag:
                if xpad_ind == 0b01:  # Short X-PAD
                    if xpad_len < 1:
                        return
                    ci_type = xpad[0] & 0x1F
                    if ci_type != 0x00:
                        cis_len = 1
                        cis.append(_XpadCi(length=3, ci_type=ci_type))

                elif xpad_ind == 0b10:  # Variable X-PAD
                    for i in range(4):
                        if xpad_len < i + 1:
                            return
                        ci_raw = xpad[i]
                        if (ci_raw & 0x1F) == 0x00:
                            break
                        cis.append(_XpadCi.from_raw(ci_raw))
                    cis_len = i + 1
            else:
                if xpad_ind in (0b01, 0b10) and prev_ci.type != -1:
                    cis_len = 0
                    cis.append(prev_ci)

        if not cis:
            return

        # Validate announced X-PAD length
        announced_len = cis_len
        for ci in cis:
            announced_len += ci.len
        if announced_len > xpad_len:
            return

        # Process CI data subfields
        offset = cis_len
        ci_type_continued = -1

        for ci in cis:
            dgli_len = self._dgli_decoder.get_len()
            subfield = xpad[offset : offset + ci.len]

            if ci.type == 1:
                # DGLI
                self._dgli_decoder.process_subfield(ci_flag, subfield)
                ci_type_continued = 1

            elif ci.type in (2, 3):
                # DLS start (2) or continuation (3)
                if self._dl_decoder.process_subfield(ci.type == 2, subfield):
                    self.dynamic_label = self._dl_decoder.label
                    self.label_changed = True
                ci_type_continued = 3

            elif self._mot_app_type != -1 and ci.type in (
                self._mot_app_type,
                self._mot_app_type + 1,
            ):
                # MOT start or continuation
                start = ci.type == self._mot_app_type
                if start:
                    self._mot_decoder.set_len(dgli_len)
                if self._mot_decoder.process_subfield(start, subfield):
                    dg = self._mot_decoder.get_data_group()
                    if self._mot_manager.handle_data_group(dg):
                        new_file = self._mot_manager.get_file()
                        if new_file is not None and self._is_displayable(new_file):
                            self.slide = new_file
                            self.slide_changed = True
                ci_type_continued = self._mot_app_type + 1

            offset += ci.len

        # Remember last CI for continuation frames
        self._last_xpad_ci = _XpadCi(length=offset, ci_type=ci_type_continued)

    def _is_displayable(self, f: MOTFile) -> bool:
        return f.content_type == _CONTENT_TYPE_IMAGE and f.content_sub_type in (
            _CONTENT_SUB_TYPE_JFIF,
            _CONTENT_SUB_TYPE_PNG,
        )
