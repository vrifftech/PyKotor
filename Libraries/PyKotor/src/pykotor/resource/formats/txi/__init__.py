from __future__ import annotations
from .txi_data import (
    TXIBaseInformation,
    TXIFontInformation,
    TXIMaterialInformation,
    TXITextureInformation,
)

from .txi_document import TXI, TXIRecord, TXISettings
from .txi_auto import read_txi, write_txi, bytes_txi
