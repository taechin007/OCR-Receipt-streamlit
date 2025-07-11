import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import tempfile
import re
import cv2
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import json
import os
from datetime import datetime


reader = easyocr.Reader(['th','en'], gpu=False)

st.set_page_config(page_title="Th-En receipt OCR Extractor", layout="wide")
st.title("Th-En receipt OCR Extractor")
st.markdown("Upload PDF or image files of receipts/invoices to extract key fields.")

proximity_ratio = st.sidebar.slider("Proximity Threshold (% of image diagonal)", min_value=1, max_value=20, value=5) / 100.0

def get_center(bbox):
    x = sum([p[0] for p in bbox]) / 4
    y = sum([p[1] for p in bbox]) / 4
    return x, y

def distance_between_boxes(box1, box2):
    x1, y1 = get_center(box1)
    x2, y2 = get_center(box2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def extract_fields_from_lines(text_lines_with_bbox, highlight_boxes, proximity_threshold):
    tax_id_pattern = re.compile(r'\b0\d{12}\b')
    date_pattern = re.compile(r'\b(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b')
    two_decimal_pattern = re.compile(r'(?:0|[1-9]\d{0,2}(?:\s*,\s*\d{3})*|\d{1,3}(?:\s*,\s*\d{3}){1,2})[.]\s*\d{2}')
    company_pattern = re.compile(r'(บริษัท[^\n]*?จำกัด)')
    doc_number_pattern = re.compile(r'\b(?=.*[0-9])(?=.*[A-Z0-9])[A-Z0-9/-]{5,}\b', re.IGNORECASE)

    fields = {
        'Tax ID': [],
        'Date': None,
        'Two Decimal Numbers': [],
        'Company Names': [],
        'Document Numbers': []
    }

    keyword_boxes = {'date': [], 'doc': []}

    for bbox, text in text_lines_with_bbox:
        if 'วันที่' in text or 'Date' in text:
            keyword_boxes['date'].append((bbox, text))
            highlight_boxes.append((bbox, "yellow", "DATE_KEY"))
        if 'เลขที่' in text:
            keyword_boxes['doc'].append((bbox, text))
            highlight_boxes.append((bbox, "yellow", "DOC_KEY"))

    for bbox, text in text_lines_with_bbox:
        tax_id_matches = re.findall(tax_id_pattern, text)
        for match in tax_id_matches:
            if match not in fields['Tax ID']:
                fields['Tax ID'].append(match)

        company_matches = re.findall(company_pattern, text)
        for match in company_matches:
            if match not in fields['Company Names']:
                fields['Company Names'].append(match.strip())

        two_decimal_matches = two_decimal_pattern.findall(text)
        if two_decimal_matches:
            cleaned_matches = []
            for match in two_decimal_matches:
                cleaned_match = re.sub(r'\s+', '', match).replace(',', '.')
                if cleaned_match.count('.') > 1:
                    cleaned_match = cleaned_match.replace('.', ',', 1).replace(',', '')
                cleaned_matches.append(cleaned_match)
            try:
                fields['Two Decimal Numbers'].extend([float(m) for m in cleaned_matches])
            except:
                pass

    for target_bbox, target_text in text_lines_with_bbox:
        for key_bbox, _ in keyword_boxes['doc']:
            dist = distance_between_boxes(target_bbox, key_bbox)
            if dist < proximity_threshold:
                match = doc_number_pattern.search(target_text)
                if match:
                    candidate = match.group().strip()
                    if candidate not in fields['Document Numbers']:
                        fields['Document Numbers'].append(candidate)
                        highlight_boxes.append((target_bbox, "blue", f"DOC\n{int(dist)}px"))

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

    fields['Two Decimal Numbers'].sort()
    if fields['Two Decimal Numbers']:
        hi = fields['Two Decimal Numbers'][-1]
        before = round(hi / 1.07, 2)
        vat = round(hi - before, 2)
        fields['Total Before VAT'] = before
        fields['VAT Amount'] = vat
        fields['Total Included VAT'] = hi

    return fields

uploaded_file = st.file_uploader("📤 Upload receipt image or PDF", type=["jpg", "png", "jpeg", "pdf"])

if uploaded_file:
    with st.spinner("⏳ Processing file..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        base_filename = os.path.splitext(uploaded_file.name)[0]
        images = convert_from_path(tmp_path, dpi=300) if uploaded_file.name.lower().endswith(".pdf") else [Image.open(tmp_path).convert("RGB")]

        for i, img in enumerate(images):
            st.subheader(f"🖼️ Page {i+1}")
            width, height = img.size
            diag = (width**2 + height**2) ** 0.5
            proximity_threshold_px = diag * proximity_ratio

            ocr_result = reader.readtext(np.array(img), detail=1, paragraph=False)
            lines = [(b, t) for (b, t, c) in ocr_result]
            full_text = "\n".join([t for (_, t) in lines])
            st.text_area("🔍 Full OCR Text", full_text, height=200)

            highlights = []
            fields = extract_fields_from_lines(lines, highlights, proximity_threshold_px)
            st.json(fields)

            draw = ImageDraw.Draw(img)
            for box, color, label in highlights:
                draw.polygon(box, outline=color, width=3)
                cx, cy = get_center(box)
                draw.text((cx, cy), label, fill=color)

            st.image(img, caption="📍 OCR Highlights", use_container_width=True)
