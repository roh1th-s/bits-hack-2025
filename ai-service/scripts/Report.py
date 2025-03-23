import os
import io
import uuid
import json
import datetime
import tempfile
import time
import shutil
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.platypus import PageBreak, KeepTogether, HRFlowable, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
import sys
from PIL.ExifTags import TAGS, GPSTAGS
import c2pa
import hashlib

def generate_case_number_with_checksum():
    """Generate a unique case number with date prefix and SHA-256 checksum"""
    date_prefix = datetime.datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4().hex)[:6].upper()
    case_number = f"{date_prefix}-{unique_id}"
    # Generate SHA-256 checksum of the case number
    checksum = hashlib.sha256(case_number.encode()).hexdigest()[:8]
    return case_number, checksum, "SHA-256"

# Try importing exiftool; if not available, use a fallback
try:
    import exiftool
except ImportError:
    print("PyExifTool not installed. Install with: pip install pyexiftool")
    def get_exif_fallback(image_path, return_data=False):
        """Fallback function when exiftool is not available"""
        try:
            with Image.open(image_path) as img:
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        exif_data = {"EXIF": exif}
                if return_data:
                    return exif_data
                else:
                    print(exif_data)
        except Exception as e:
            if return_data:
                return {"Error": str(e)}
            else:
                print(f"Error extracting EXIF data: {str(e)}")

try:
    import c2pa
except ImportError:
    c2pa = None

def get_c2pa_data(image_path):
    """Extract C2PA metadata if available"""
    c2pa_data = {}
    try:
        if c2pa:
            reader = c2pa.Reader.from_file(image_path)
            if reader.manifest_store:
                c2pa_data = json.loads(reader.json())
                # Remove large binary data that can't be serialized
                c2pa_data.pop('signature', None)
                c2pa_data.pop('cert_chain', None)
            else:
                c2pa_data = {"status": "No C2PA manifest found"}
        else:
            c2pa_data = {"error": "c2pa library not installed - install with: pip install c2pa"}
    except Exception as err:
        c2pa_data = {"error": str(err)}
    return c2pa_data

def get_pil_exif(image_path):
    """Extract EXIF metadata using PIL with detailed tag processing"""
    exif_data = {}
    try:
        with Image.open(image_path) as img:
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif_info = img._getexif()
                for tag_id, value in exif_info.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    exif_data[tag_name] = value
                if 'GPSInfo' in exif_data:
                    gps_data = {}
                    for gps_tag in exif_data['GPSInfo'].keys():
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        gps_data[gps_tag_name] = exif_data['GPSInfo'][gps_tag]
                    exif_data['GPSInfo'] = gps_data
                exif_data['Image Information'] = {
                    'Format': img.format,
                    'Mode': img.mode,
                    'Size': img.size,
                    'Width': img.width,
                    'Height': img.height,
                    'Animated': getattr(img, 'is_animated', False),
                    'Frames': getattr(img, 'n_frames', 1)
                }
    except Exception as e:
        exif_data['Error'] = f"EXIF extraction error: {str(e)}"
    return exif_data

class HeatmapImage(Flowable):
    """A flowable for adding heatmap images with captions"""
    def __init__(self, image_path, width=6*inch, height=None, caption=""):
        Flowable.__init__(self)
        self.img = Image.open(image_path)
        self.caption = caption
        self.width = width
        if height is None:
            self.height = width * self.img.height / self.img.width
        else:
            self.height = height
            
    def draw(self):
        self.canv.drawImage(self.img, 0, 12, width=self.width, height=self.height, preserveAspectRatio=True)
        if self.caption:
            self.canv.setFont("Helvetica", 9)
            self.canv.drawCentredString(self.width/2, 0, self.caption)

def generate_case_number():
    """Generate a unique case number with date prefix"""
    date_prefix = datetime.datetime.now().strftime("%Y%m%d")
    unique_id = str(uuid.uuid4().hex)[:6].upper()
    return f"{date_prefix}-{unique_id}"

def format_metadata_table(metadata):
    """Format the metadata as a table for the report"""
    if not metadata:
        return [["No metadata available", ""]]
    formatted_data = []
    if isinstance(metadata, str):
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            return [["Raw Metadata", metadata]]
    else:
        metadata_dict = metadata
    def flatten_dict(d, prefix=""):
        items = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, key).items())
            elif isinstance(v, list) and all(isinstance(x, dict) for x in v):
                for i, item in enumerate(v):
                    items.extend(flatten_dict(item, f"{key}[{i}]").items())
            else:
                items.append((key, v))
        return dict(items)
    if isinstance(metadata_dict, dict):
        flat_dict = flatten_dict(metadata_dict)
        formatted_data = [[k, str(v)] for k, v in flat_dict.items()]
    else:
        formatted_data = [["Raw Metadata", str(metadata_dict)]]
    return formatted_data

def create_header_footer(canvas, doc, case_number, investigator_name, checksum=None, checksum_algorithm=None):
    """Create header and footer for each page"""
    canvas.saveState()
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(inch, 10.5 * inch, f"CASE #{case_number}")
    if checksum and checksum_algorithm:
        canvas.setFont('Helvetica', 8)
        canvas.drawString(inch, 10.35 * inch, f"Checksum ({checksum_algorithm}): {checksum}")
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, 10.3 * inch, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    canvas.drawString(inch, 10.1 * inch, f"Investigator: {investigator_name}")
    canvas.line(inch, 10*inch, 7.5*inch, 10*inch)
    canvas.setFont('Helvetica', 8)
    canvas.line(inch, 0.75*inch, 7.5*inch, 0.75*inch)
    canvas.drawString(inch, 0.5 * inch, "CONFIDENTIAL - FORENSIC INVESTIGATION REPORT")
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()

def get_exif_data(image_path, return_data=False):
    """
    Extracts EXIF metadata from the given image file.
    """
    try:
        if 'exiftool' in sys.modules:
            with exiftool.ExifTool() as et:
                metadata = et.execute_json("-j", image_path)
            if metadata:
                if return_data:
                    return metadata[0]
                else:
                    print(json.dumps(metadata[0], indent=4))
            else:
                if return_data:
                    return {}
                else:
                    print("No EXIF data found.")
        else:
            return get_exif_fallback(image_path, return_data)
    except Exception as e:
        if return_data:
            return {"Error": str(e)}
        else:
            print(f"Error extracting EXIF data: {str(e)}")
    if return_data:
        return {}

def format_report_text(text):
    """Format text with basic markdown parsing for better PDF rendering"""
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'(\d+)\.\s+', r'<bullet>\1.</bullet> ', text)
    return text

def generate_pdf_report(output_path, case_number, investigator_name, analysis_results, image_path, grad_cam_path=None):
    """
    Generate a PDF report for profile analysis based on multi-modal evaluation results.
    This report includes case summary, media preview, forensic metadata, and detailed
    evaluation results for both text and image modalities, including GradCAM and LIME visualizations.
    """
    if not case_number:
        case_number, checksum, checksum_algorithm = generate_case_number_with_checksum()
    else:
        checksum = hashlib.sha256(case_number.encode()).hexdigest()[:8]
        checksum_algorithm = "SHA-256"
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=1.5*inch,
        bottomMargin=inch
    )
    
    styles = getSampleStyleSheet()
    custom_styles = {
        'CaseHeader': {
            'parent': 'Heading1',
            'fontSize': 16,
            'alignment': TA_CENTER,
            'spaceAfter': 12
        },
        'SectionTitle': {
            'parent': 'Heading2',
            'fontSize': 14,
            'spaceAfter': 6,
            'spaceBefore': 12,
            'textColor': colors.darkblue
        },
        'SubSectionTitle': {
            'parent': 'Heading3',
            'fontSize': 12,
            'spaceAfter': 6,
            'textColor': colors.darkblue
        },
        'CellText': {
            'parent': 'Normal',
            'fontSize': 9
        },
        'Note': {
            'parent': 'Italic',
            'fontSize': 9,
            'textColor': colors.gray
        },
        'Verdict': {
            'parent': 'Normal',
            'fontSize': 12,
            'alignment': TA_CENTER,
            'textColor': colors.white,
            'backColor': colors.red,
            'borderPadding': 8,
            'borderWidth': 1,
            'borderColor': colors.black,
            'borderRadius': 8
        }
    }
    for style_name, properties in custom_styles.items():
        parent_style = properties.pop('parent', 'Normal')
        styles.add(ParagraphStyle(name=style_name, parent=styles[parent_style], **properties))
    styles.add(ParagraphStyle(name='VerdictReal', parent=styles['Verdict'], backColor=colors.green))
    styles['Normal'].fontSize = 10
    styles['Normal'].spaceBefore = 6
    styles['Normal'].leading = 14
    if "Code" in styles:
        styles["Code"].fontSize = 8
        styles["Code"].leading = 10
    
    story = []
    
    # Title Section
    story.append(Paragraph("PROFILE ANALYSIS REPORT", styles['CaseHeader']))
    story.append(Spacer(1, 0.25*inch))
    
    # Case Summary
    data = [
        ["Case Number:", case_number],
        ["Case Checksum:", f"{checksum} ({checksum_algorithm})"],
        ["Investigation Date:", datetime.datetime.now().strftime('%Y-%m-%d')],
        ["Investigator:", investigator_name],
        ["Analysis Method:", "Multi-modal Profile Analysis"]
    ]
    case_table = Table(data, colWidths=[2*inch, 4*inch])
    case_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(case_table)
    story.append(Spacer(1, 0.25*inch))
    
    # Media Preview
    try:
        story.append(Paragraph("MEDIA ANALYZED", styles['SectionTitle']))
        img = Image.open(image_path)
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        display_width = 5 * inch
        display_height = display_width * aspect
        if display_height > 4 * inch:
            display_height = 4 * inch
            display_width = display_height / aspect
        img_obj = RLImage(image_path, width=display_width, height=display_height)
        story.append(img_obj)
        story.append(Paragraph(f"Filename: {os.path.basename(image_path)}<br/>Dimensions: {img_width}x{img_height} pixels", styles['Note']))
        story.append(Spacer(1, 0.25*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    except Exception as e:
        story.append(Paragraph(f"Error loading image: {str(e)}", styles['Note']))
    
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    
    # Digital Forensics Metadata Section
    story.append(Paragraph("DIGITAL FORENSICS METADATA", styles['SectionTitle']))
    exif_metadata = get_pil_exif(image_path)
    if exif_metadata:
        table_data = [["Metadata Field", "Value"]]
        formatted_metadata = format_metadata_table(exif_metadata)
        table_data.extend(formatted_metadata)
        metadata_table = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
        ]))
        story.append(KeepTogether([
            Paragraph("Technical Metadata Extraction", styles['SubSectionTitle']),
            metadata_table,
            Spacer(1, 0.25*inch)
        ]))
    else:
        story.append(Paragraph("No technical metadata could be extracted", styles['Note']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    
    # Content Credentials (C2PA)
    story.append(Paragraph("CONTENT CREDENTIALS (C2PA)", styles['SectionTitle']))
    c2pa_data = get_c2pa_data(image_path)
    if c2pa_data:
        table_data = [["C2PA Field", "Value"]]
        formatted_c2pa = format_metadata_table(c2pa_data)
        table_data.extend(formatted_c2pa)
        c2pa_table = Table(table_data, colWidths=[2.5*inch, 4.5*inch])
        c2pa_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
        ]))
        story.append(KeepTogether([
            Paragraph("Content Authenticity Verification", styles['SubSectionTitle']),
            c2pa_table,
            Spacer(1, 0.25*inch)
        ]))
        if 'error' not in c2pa_data:
            validation_status = "Valid C2PA Signature Found" if c2pa_data.get('signature_valid', False) else "Invalid or Missing C2PA Signature"
            story.append(Paragraph(f"Validation Status: {validation_status}", styles['Note']))
        else:
            story.append(Paragraph("No C2PA content credentials found", styles['Note']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    
    # Analysis Results Section
    story.append(Paragraph("ANALYSIS RESULTS", styles['SectionTitle']))
    # Expecting analysis_results to be a dictionary with keys: text_evaluation, image_evaluation, final_report
    if isinstance(analysis_results, str):
        try:
            results_dict = json.loads(analysis_results)
        except json.JSONDecodeError:
            results_dict = {"Error": "Invalid JSON format in analysis results"}
    else:
        results_dict = analysis_results
    
    # Text Evaluation Subsection
    if "text_evaluation" in results_dict:
        text_eval = results_dict["text_evaluation"]
        story.append(Paragraph("Text Evaluation", styles['SubSectionTitle']))
        story.append(Paragraph(f"Score: {text_eval.get('score', 'N/A')}", styles['Normal']))
        story.append(Paragraph(f"Prediction: {text_eval.get('prediction', 'N/A')}", styles['Normal']))
        story.append(Paragraph("Extracted Profile Text:", styles['Normal']))
        story.append(Paragraph(text_eval.get("text", ""), styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Image Evaluation Subsection
    if "image_evaluation" in results_dict:
        img_eval = results_dict["image_evaluation"]
        story.append(Paragraph("Image Evaluation", styles['SubSectionTitle']))
        story.append(Paragraph(f"Average Deepfake Score: {img_eval.get('average_deepfake_score', 'N/A')}", styles['Normal']))
        # Optionally list individual results
        ind_results = img_eval.get("individual_results", [])
        if ind_results:
            table_data = [["Image URL", "Deepfake Score", "GradCAM", "LIME"]]
            for entry in ind_results:
                # Try to extract deepfake score; fallback if unavailable
                try:
                    df_score = entry["deepfake"][1]["score"]
                except Exception:
                    df_score = "N/A"
                gradcam_str = "Available" if entry.get("gradcam") else "N/A"
                lime_str = "Available" if entry.get("lime") else "N/A"
                table_data.append([entry.get("url", "N/A"), df_score, gradcam_str, lime_str])
            img_table = Table(table_data, colWidths=[2.5*inch, 1*inch, 1.5*inch, 1.5*inch])
            img_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(img_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Final Overall Verdict
    if "final_report" in results_dict:
        final_eval = results_dict["final_report"]
        verdict = final_eval.get("verdict", "N/A")
        final_score = final_eval.get("final_score", "N/A")
        verdict_style = 'Verdict' if "Fake" in str(verdict) else 'VerdictReal'
        story.append(Paragraph(f"FINAL VERDICT: {verdict} (Score: {final_score})", styles[verdict_style]))
        story.append(Spacer(1, 0.2*inch))
    
    # Visual Evidence Section (for GradCAM and LIME)
    story.append(Paragraph("VISUAL EVIDENCE", styles['SubSectionTitle']))
    story.append(Spacer(1, 0.1*inch))
    # Attempt to retrieve representative GradCAM and LIME outputs from image_evaluation individual results
    representative_gradcam = None
    representative_lime = None
    if "image_evaluation" in results_dict and "individual_results" in results_dict["image_evaluation"]:
        for res in results_dict["image_evaluation"]["individual_results"]:
            if not representative_gradcam and res.get("gradcam"):
                representative_gradcam = res["gradcam"]
            if not representative_lime and res.get("lime"):
                representative_lime = res["lime"]
            if representative_gradcam and representative_lime:
                break
    lime_image = None
    gradcam_image = None
    png_paths = []
    import base64
    # Process LIME visualization if available (expects a dict with keys like "overlay" or "saliency")
    if representative_lime and isinstance(representative_lime, dict):
        for field in ["overlay", "saliency"]:
            if representative_lime.get(field):
                try:
                    base64_data = representative_lime[field]
                    if base64_data.startswith('data:image'):
                        base64_data = base64_data.split(',', 1)[1]
                    binary_data = base64.b64decode(base64_data)
                    lime_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    lime_file.write(binary_data)
                    lime_file.close()
                    lime_image = lime_file.name
                    print(f"Successfully created temporary LIME visualization file: {lime_image}")
                    break
                except Exception as e:
                    print(f"Error processing LIME {field} image: {e}")
    # Process GradCAM visualization similarly
    if representative_gradcam and isinstance(representative_gradcam, dict):
        for field in ["overlay", "saliency"]:
            if representative_gradcam.get(field):
                try:
                    base64_data = representative_gradcam[field]
                    if base64_data.startswith('data:image'):
                        base64_data = base64_data.split(',', 1)[1]
                    binary_data = base64.b64decode(base64_data)
                    gradcam_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    gradcam_file.write(binary_data)
                    gradcam_file.close()
                    gradcam_image = gradcam_file.name
                    print(f"Successfully created temporary GradCAM visualization file: {gradcam_image}")
                    break
                except Exception as e:
                    print(f"Error processing GradCAM {field} image: {e}")
    # Add LIME visualization if available
    if lime_image and os.path.exists(lime_image):
        try:
            img = Image.open(lime_image)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 5 * inch
            display_height = display_width * aspect
            if display_height > 4 * inch:
                display_height = 4 * inch
                display_width = display_height / aspect
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            lime_png_path = lime_image.rsplit('.', 1)[0] + "_lime.png"
            img.save(lime_png_path, "PNG")
            png_paths.append(lime_png_path)
            story.append(Paragraph("LIME Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Regions highlighted by LIME influencing the detection decision.", styles['Note']))
            img_obj = RLImage(lime_png_path, width=display_width, height=display_height)
            story.append(img_obj)
            story.append(Spacer(1, 0.5*inch))
        except Exception as e:
            story.append(Paragraph(f"Error loading LIME visualization image: {str(e)}", styles['Note']))
            print(f"Exception when adding LIME visualization to PDF: {e}")
    # Add GradCAM visualization if available
    if gradcam_image and os.path.exists(gradcam_image):
        try:
            img = Image.open(gradcam_image)
            img_width, img_height = img.size
            aspect = img_height / float(img_width)
            display_width = 5 * inch
            display_height = display_width * aspect
            if display_height > 4 * inch:
                display_height = 4 * inch
                display_width = display_height / aspect
            if img.mode not in ['RGB', 'RGBA']:
                img = img.convert('RGB')
            gradcam_png_path = gradcam_image.rsplit('.', 1)[0] + "_gradcam.png"
            img.save(gradcam_png_path, "PNG")
            png_paths.append(gradcam_png_path)
            if display_height > 3 * inch:
                story.append(PageBreak())
            story.append(Paragraph("GradCAM Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Regions highlighted by GradCAM indicating potential manipulations.", styles['Note']))
            img_obj = RLImage(gradcam_png_path, width=display_width, height=display_height)
            story.append(img_obj)
        except Exception as e:
            story.append(Paragraph(f"Error loading GradCAM visualization image: {str(e)}", styles['Note']))
            print(f"Exception when adding GradCAM visualization to PDF: {e}")
    
    # End of report content; build the PDF
    doc.build(story, onFirstPage=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name),
             onLaterPages=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name))
    
    # Clean up temporary visualization files
    for tmp in [lime_image, gradcam_image] + png_paths:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
                print(f"Cleaned up temporary file: {tmp}")
            except Exception as e:
                print(f"Error cleaning up temporary file {tmp}: {e}")
    
    return output_path
