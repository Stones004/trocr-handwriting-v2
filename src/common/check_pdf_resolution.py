"""
Checks whether raising PDFLoader's dpi would actually recover real detail,
or just upscale-blur the same data one step earlier in the pipeline.

For each embedded raster image in a PDF page, computes the EFFECTIVE DPI
implied by (native pixel dimensions / page size in inches). If that's
meaningfully higher than the dpi PDFLoader currently renders at (300),
there's real headroom -- raising PDFLoader's dpi to match will pull out
detail that exists in the file but is currently being discarded. If the
embedded image's native resolution is already at or below 300 DPI, this
loader is already extracting everything available -- raising dpi further
would only interpolate, and the real fix is re-scanning source documents
at higher resolution, not a code change here.

Usage:
    python check_pdf_resolution.py path/to/one_of_your_scanned.pdf
"""
import sys

import fitz  # PyMuPDF

if len(sys.argv) < 2:
    raise SystemExit("Usage: python check_pdf_resolution.py path/to.pdf")

pdf_path = sys.argv[1]
CURRENT_RENDER_DPI = 300  # matches PDFLoader's current default

doc = fitz.open(pdf_path)

print(f"File: {pdf_path}")
print(f"Pages: {len(doc)}")
print(f"Current PDFLoader render DPI: {CURRENT_RENDER_DPI}")
print("=" * 70)

# Check a handful of pages, not the whole document -- scans are usually
# consistent within one PDF, so a sample is enough to answer the question.
pages_to_check = range(2, min(len(doc), 7))  # skip first 2 like PDFLoader does

any_images_found = False

for page_num in pages_to_check:
    page = doc.load_page(page_num)
    page_rect = page.rect  # in points, 72 points = 1 inch
    page_width_in = page_rect.width / 72
    page_height_in = page_rect.height / 72

    images = page.get_images(full=True)

    if not images:
        continue

    any_images_found = True
    print(f"\nPage {page_num + 1} ({page_width_in:.2f}\" x {page_height_in:.2f}\"):")

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        native_w = base_image["width"]
        native_h = base_image["height"]

        # Effective DPI if this image covers roughly the full page --
        # a reasonable assumption for a single full-page scan.
        effective_dpi_w = native_w / page_width_in
        effective_dpi_h = native_h / page_height_in

        print(
            f"  Embedded image {img_index}: {native_w}x{native_h}px  "
            f"-> effective ~{effective_dpi_w:.0f} DPI (w) / {effective_dpi_h:.0f} DPI (h)"
        )

        headroom = min(effective_dpi_w, effective_dpi_h) / CURRENT_RENDER_DPI
        if headroom > 1.15:
            print(
                f"    -> REAL HEADROOM: source has ~{headroom:.1f}x more "
                f"detail than current {CURRENT_RENDER_DPI} DPI render is "
                f"using. Raising PDFLoader's dpi would recover genuine "
                f"detail, not just interpolate."
            )
        else:
            print(
                f"    -> NO HEADROOM: source resolution is at/near "
                f"{CURRENT_RENDER_DPI} DPI already. Raising render dpi "
                f"here would just upscale/interpolate, same problem as "
                f"before, just moved earlier in the pipeline."
            )

if not any_images_found:
    print(
        "\nNo embedded raster images found on the sampled pages -- this "
        "PDF may be vector-based (e.g. exported from a word processor) "
        "rather than a scanned document. If so, PDFLoader's rendering DPI "
        "directly controls output quality with NO ceiling from a source "
        "scan -- raising it would help without the caveat above."
    )

doc.close()