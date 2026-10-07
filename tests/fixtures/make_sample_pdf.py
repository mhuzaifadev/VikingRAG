"""Generate tests/fixtures/sample.pdf (run manually if regenerating)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

# Minimal PDF with extractable Helvetica text (offsets approximate; pypdf still parses).
_PDF = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R
   /Resources << /Font << /F1 5 0 R >> >>
>>
endobj
4 0 obj
<< /Length 95 >>
stream
BT
/F1 18 Tf
72 720 Td
(SYSTEM ARCHITECTURE) Tj
0 -40 Td
/F1 12 Tf
(PostgreSQL stores hierarchy.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000261 00000 n 
0000000406 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
479
%%EOF
"""


def main() -> None:
    out = Path(__file__).with_name("sample.pdf")
    out.write_bytes(_PDF)
    reader = PdfReader(BytesIO(_PDF))
    assert len(reader.pages) == 1
    assert "ARCHITECTURE" in (reader.pages[0].extract_text() or "")
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
