import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import tempfile
import re
import cv2
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import json
import os
from datetime import datetime
import time

reader = easyocr.Reader(['th', 'en'], gpu=True)

st.set_page_config(page_title="Th-En receipt OCR Extractor", layout="wide")
st.title("Th-En receipt OCR Extractor")
st.markdown("Upload multiple PDF or image files of receipts/invoices to extract key fields.")

proximity_ratio = st.sidebar.slider("Proximity Threshold (% of image diagonal)", 1, 20, 5) / 100.0

preprocess_option = st.sidebar.selectbox(
    "Preprocessing Method",
    [
        "None",
        "Grayscale",
        "Thresholding (Binary)",
        "Adaptive Thresholding",
        "Denoise",
        "Sharpen",
        "Grayscale + Adaptive + Denoise",
        "Grayscale + Sharpen",
        "Grayscale + Thresholding",
        "Denoise + Sharpen",
        "Grayscale + Contrast Stretching",
        "Adaptive + Denoise + Sharpen"
    ]
)

st.sidebar.markdown("""
<hr>
<div style='font-size: 10px'>

---
### ℹ️ Preprocessing Help

- **None**: Use when input images are already clean and high-contrast.
- **Grayscale**: Converts image to shades of gray. Useful for reducing color noise and focusing OCR on structure.
- **Thresholding (Binary)**: Converts to black and white using global threshold. Best for well-lit, clean documents with sharp text.
- **Adaptive Thresholding**: Dynamically binarizes regions with uneven lighting. Works well for shadowed or aged documents.
- **Denoise**: Reduces speckle noise or background textures. Helps prevent OCR misreads on degraded paper or scanned artifacts.
- **Sharpen**: Enhances edges. Useful if characters are blurry or slightly faded.
- **Grayscale + Adaptive + Denoise**: Good general-purpose combo for moderate-quality documents with lighting and noise issues.
- **Grayscale + Sharpen**: For slightly faded or low-contrast text. Improves edge clarity.
- **Grayscale + Thresholding**: Ideal for clean scans with consistent lighting. Can over-binarize noisy scans.
- **Denoise + Sharpen**: Removes artifacts and clarifies text simultaneously. Great for compressed or faxed documents.
- **Grayscale + Contrast Stretching**: Improves brightness and dynamic range. Useful for dull, washed-out scans.
- **Adaptive + Denoise + Sharpen**: Strongest option for difficult cases—blurry, low-quality, or unevenly lit scans.

</div>
""", unsafe_allow_html=True)

def preprocess_image(pil_image, method):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if method == "None":
        return pil_image
    elif method == "Grayscale":
        return Image.fromarray(gray)
    elif method == "Thresholding (Binary)":
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return Image.fromarray(thresh)
    elif method == "Adaptive Thresholding":
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 15, 10)
        return Image.fromarray(adapt)
    elif method == "Denoise":
        denoise = cv2.fastNlMeansDenoising(gray, h=30)
        return Image.fromarray(denoise)
    elif method == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        return Image.fromarray(sharpened)
    elif method == "Grayscale + Adaptive + Denoise":
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 15, 10)
        denoise = cv2.fastNlMeansDenoising(adapt, h=30)
        return Image.fromarray(denoise)
    elif method == "Grayscale + Sharpen":
        sharpen = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        return Image.fromarray(sharpen)
    elif method == "Grayscale + Thresholding":
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return Image.fromarray(thresh)
    elif method == "Denoise + Sharpen":
        denoise = cv2.fastNlMeansDenoising(gray, h=30)
        sharpen = cv2.filter2D(denoise, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        return Image.fromarray(sharpen)
    elif method == "Grayscale + Contrast Stretching":
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return Image.fromarray(norm)
    elif method == "Adaptive + Denoise + Sharpen":
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 15, 10)
        denoise = cv2.fastNlMeansDenoising(adapt, h=30)
        sharpen = cv2.filter2D(denoise, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        return Image.fromarray(sharpen)
    return pil_image

def get_center(bbox):
    x = float(sum([p[0] for p in bbox]) / 4)
    y = float(sum([p[1] for p in bbox]) / 4)
    return x, y

def distance_between_boxes(box1, box2):
    x1, y1 = get_center(box1)
    x2, y2 = get_center(box2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def convert_bbox_to_list(bbox):
    return [[float(coord[0]), float(coord[1])] for coord in bbox]

def extract_fields_from_lines(text_lines_with_bbox, highlight_boxes, proximity_threshold):
    tax_id_pattern = re.compile(r'\b0[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d[\s-]?\d\b')
    date_pattern = re.compile(r'\b\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4}|\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4}\b')
    two_decimal_pattern = re.compile(r'\b(?:0|[1-9]\d{0,2}(?:\s*,\s*\d{3})*|\d{1,3}(?:\s*,\s*\d{3}){1,2})[.,]+\s*\d{2}\b')
    company_pattern = re.compile(r'บริษัท[\s\S]{0,100}?จ[าาำ]กัด')
    doc_number_pattern = re.compile(r'\b(?=.*[0-9])(?=.*[A-Z0-9])[A-Z0-9/-]{5,}\b', re.IGNORECASE)

    fields = {
        'Tax ID': [], 'Date': None, 'Two Decimal Numbers': [],
        'Company Names': [], 'Document Numbers': [], 'Bounding Boxes': []
    }

    keyword_boxes = {'date': [], 'doc': []}

    for bbox, text in text_lines_with_bbox:
        if 'วันที่' in text or 'Date' in text or 'วันที' in text or 'date' in text:  
            keyword_boxes['date'].append((bbox, text))
            highlight_boxes.append((bbox, "yellow", "DATE_KEY"))
        if 'เลขที่' in text or 'เลบที่' in text or 'No.' in text or 'no.' in text or 'เกทที่' in text:
            keyword_boxes['doc'].append((bbox, text))
            highlight_boxes.append((bbox, "yellow", "DOC_KEY"))

    for bbox, text in text_lines_with_bbox:
        for match in re.findall(tax_id_pattern, text):
            if match not in fields['Tax ID']:
                fields['Tax ID'].append(match)
                fields['Bounding Boxes'].append({'field': 'Tax ID', 'value': match, 'bbox': convert_bbox_to_list(bbox)})

        for match in re.findall(company_pattern, text):
            if match not in fields['Company Names']:
                fields['Company Names'].append(match.strip())
                fields['Bounding Boxes'].append({'field': 'Company Name', 'value': match.strip(), 'bbox': convert_bbox_to_list(bbox)})

        for match in re.findall(two_decimal_pattern, text):
            digits_only = re.sub(r'[^\d]', '', match)
            if len(digits_only) > 2:
                reconstructed = digits_only[:-2] + '.' + digits_only[-2:]
                try:
                    val = float(reconstructed)
                    fields['Two Decimal Numbers'].append(val)
                    fields['Bounding Boxes'].append({'field': 'Decimal', 'value': val, 'bbox': convert_bbox_to_list(bbox)})
                except:
                    continue

    for target_bbox, target_text in text_lines_with_bbox:
        for key_bbox, _ in keyword_boxes['doc']:
            dist = distance_between_boxes(target_bbox, key_bbox)
            if dist < proximity_threshold:
                match = doc_number_pattern.search(target_text)
                if match:
                    val = match.group().strip()
                    if val not in fields['Document Numbers']:
                        fields['Document Numbers'].append(val)
                        highlight_boxes.append((target_bbox, "blue", f"DOC\n{int(dist)}px"))
                        fields['Bounding Boxes'].append({'field': 'Document Number', 'value': val, 'bbox': convert_bbox_to_list(target_bbox)})

        for key_bbox, _ in keyword_boxes['date']:
            dist = distance_between_boxes(target_bbox, key_bbox)
            if dist < proximity_threshold:
                match = date_pattern.search(target_text)
                if match:
                    date_str = match.group()
                    try:
                        dt = datetime.strptime(re.sub(r'[/.-]', '/', date_str), '%d/%m/%Y')
                    except ValueError:
                        try:
                            dt = datetime.strptime(re.sub(r'[/.-]', '/', date_str), '%Y/%m/%d')
                        except:
                            continue
                    if not fields['Date']:
                        fields['Date'] = dt.strftime('%d/%m/%Y')
                        highlight_boxes.append((target_bbox, "green", f"DATE\n{int(dist)}px"))
                        fields['Bounding Boxes'].append({'field': 'Date', 'value': fields['Date'], 'bbox': convert_bbox_to_list(target_bbox)})

    fields['Two Decimal Numbers'].sort()
    decimals = fields['Two Decimal Numbers']

    def approx_in(value, collection, tol=0.01):
        return any(abs(val - value) <= tol for val in collection)

    def compute_vat_fields(candidate_hi):
        before = round(candidate_hi / 1.07, 2)
        vat = round(candidate_hi - before, 2)
        if approx_in(before, decimals) and approx_in(vat, decimals):
            return before, vat, candidate_hi
        return None

    valid_result = None
    for hi_candidate in reversed(decimals):
        result = compute_vat_fields(hi_candidate)
        if result:
            valid_result = result
            break

    if valid_result:
        before, vat, hi = valid_result
        fields['Total Before VAT'] = before
        fields['VAT Amount'] = vat
        fields['Total Included VAT'] = hi
    else:
        fields.pop('Total Before VAT', None)
        fields.pop('VAT Amount', None)
        fields.pop('Total Included VAT', None)

    return fields

uploaded_files = st.file_uploader("📤 Upload receipt images or PDFs", type=["jpg", "png", "jpeg", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("▶️ Start OCR"):
        all_results = []
        st.markdown("---")
        st.subheader("📥 Download Extracted Results")

        start_time = time.time()
        for uploaded_file in uploaded_files:
            st.markdown(f"### 📄 `{uploaded_file.name}`")
            with st.spinner("⏳ Processing..."):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                images = convert_from_path(tmp_path, dpi=300) if uploaded_file.name.lower().endswith(".pdf") else [Image.open(tmp_path).convert("RGB")]
                images = [preprocess_image(img, preprocess_option) for img in images]

                for i, img in enumerate(images):
                    width, height = img.size
                    diag = (width**2 + height**2) ** 0.5
                    threshold_px = diag * proximity_ratio

                    ocr_result = reader.readtext(np.array(img), detail=1, paragraph=False)
                    lines = [(b, t) for (b, t, c) in ocr_result]
                    full_text = "\n".join([t for (_, t) in lines])

                    highlights = []
                    fields = extract_fields_from_lines(lines, highlights, threshold_px)
                    fields["File Name"] = uploaded_file.name
                    fields["Page"] = i + 1
                    fields["Full Text"] = full_text
                    all_results.append(fields)

                    draw = ImageDraw.Draw(img)
                    for box, color, label in highlights:
                        draw.polygon(box, outline=color, width=3)
                        cx, cy = get_center(box)
                        draw.text((cx, cy), label, fill=color)

                    thumb = img.copy()
                    thumb.thumbnail((300, 300))
                    st.image(thumb, caption=f"🖼️ Page {i+1} Thumbnail", use_container_width=False)

                    with st.expander(f"📄 Page {i+1} - Full View"):
                        st.image(img, caption="🖼️ Full Annotated Image", use_container_width=True)
                        st.text_area("🔍 Full OCR Text", full_text, height=200)
                        st.json(fields)

        duration = time.time() - start_time
        st.success(f"✅ Processed {len(uploaded_files)} file(s) in {duration:.2f} seconds")

        df_export = pd.json_normalize(all_results)
        json_str = json.dumps(all_results, ensure_ascii=False, indent=2, default=str)

        st.download_button("⬇️ Download JSON", data=json_str, file_name="ocr_results.json", mime="application/json", key="json")
        excel_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        df_export.to_excel(excel_tmp.name, index=False)
        with open(excel_tmp.name, "rb") as f:
            st.download_button("⬇️ Download Excel", data=f.read(), file_name="ocr_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="excel")
