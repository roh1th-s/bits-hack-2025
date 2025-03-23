from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image
import os
import tempfile
import datetime
import json
from scripts.Report import  generate_case_number

def generate_video_pdf_report(output_path, case_number, investigator_name, video_analysis, video_path=None, sample_frames=None, model_path=None):
    # Ensure we have a case number
    if not case_number:
        case_number = generate_case_number()

    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=1.5*inch,
        bottomMargin=inch
    )

    # Styles
    styles = getSampleStyleSheet()

    # Define custom styles
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

    # Add the custom styles to the stylesheet
    for style_name, properties in custom_styles.items():
        parent_style = properties.pop('parent', 'Normal')
        styles.add(ParagraphStyle(name=style_name, parent=styles[parent_style], **properties))

    # Add VerdictReal as a variant of Verdict
    styles.add(ParagraphStyle(name='VerdictReal', parent=styles['Verdict'], backColor=colors.green))

    # Modify existing styles
    styles['Normal'].fontSize = 10
    styles['Normal'].spaceBefore = 6
    styles['Normal'].leading = 14
    styles["Code"].fontSize = 8
    styles["Code"].leading = 10

    # Story (content elements)
    story = []

    # Title
    story.append(Paragraph(f"VIDEO DEEPFAKE ANALYSIS REPORT", styles['CaseHeader']))
    story.append(Spacer(1, 0.25*inch))

    # Case summary
    data = [
        ["Case Number:", case_number],
        ["Investigation Date:", datetime.datetime.now().strftime('%Y-%m-%d')],
        ["Investigator:", investigator_name],
        ["Analysis Method:", "Multi-modal Deepfake Detection & Forensic Analysis"]
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

    # Video Preview
    if video_path:
        story.append(Paragraph("VIDEO ANALYZED", styles['SectionTitle']))
        story.append(Paragraph(f"Filename: {os.path.basename(video_path)}", styles['Note']))
        story.append(Spacer(1, 0.25*inch))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))

    # Analysis Results
    story.append(Paragraph("ANALYSIS RESULTS", styles['SectionTitle']))

    # Extract the relevant parts from the video_analysis
    if isinstance(video_analysis, str):
        try:
            results_dict = json.loads(video_analysis)
        except json.JSONDecodeError:
            results_dict = {"Error": "Invalid JSON format in analysis results"}
    else:
        results_dict = video_analysis

    # Add summary verdict first (most important)
    if "overall_assessment" in results_dict:
        overall_assessment = results_dict["overall_assessment"]
        is_fake = overall_assessment.get("is_likely_fake", False)

        verdict_style = 'Verdict' if is_fake else 'VerdictReal'
        verdict_text = f"VERDICT: {'Fake' if is_fake else 'Real'}"

        story.append(Paragraph(verdict_text, styles[verdict_style]))
        story.append(Spacer(1, 0.2*inch))

    # Add detailed analysis
    if "video_analysis" in results_dict:
        video_analysis = results_dict["video_analysis"]

        # Frame-by-Frame Analysis Table
        story.append(Paragraph("Frame-by-Frame Analysis", styles['SubSectionTitle']))
        frame_table_data = [["Frame #", "Timestamp", "Result", "Score", "Status"]]

        for frame in video_analysis.get("results", []):
            frame_table_data.append([
                frame["frame_number"],
                frame["timestamp"],
                frame["prediction"],
                f"{frame['fusion_score']:.4f}" if frame["fusion_score"] is not None else "N/A",
                "Suspicious" if frame["is_fake"] else "OK"
            ])

        frame_table = Table(frame_table_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        frame_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (0, 0), colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(frame_table)
        story.append(Spacer(1, 0.25*inch))

        # Per-Frame Suspicion Rate Table - Adding more detailed analysis
        story.append(Paragraph("Frame-by-Frame Suspicion Rate Analysis", styles['SubSectionTitle']))
        
        # Get frame scores from the video analysis
        frame_scores = []
        if "video_analysis" in results_dict and "results" in results_dict["video_analysis"]:
            for frame in results_dict["video_analysis"]["results"]:
                if "fusion_score" in frame and frame["fusion_score"] is not None:
                    # Normalize the score between 0-1 if needed
                    score = abs(frame["fusion_score"]) if frame["fusion_score"] < 0 else frame["fusion_score"]
                    frame_scores.append(score)
        
        # If frame_scores is not populated from the above, check if it's directly in the results
        if not frame_scores and "frame_scores" in results_dict:
            frame_scores = results_dict["frame_scores"]
        
        if frame_scores:
            # Create the table data
            suspicion_table_data = [["Frame #", "Suspicion Score", "Assessment"]]
            
            # Add data for each frame, limit to a reasonable number to avoid overly long tables
            max_frames_to_show = min(len(frame_scores), 20)
            frame_interval = max(1, len(frame_scores) // max_frames_to_show)
            
            for i in range(0, len(frame_scores), frame_interval):
                if len(suspicion_table_data) > max_frames_to_show:
                    break
                    
                score = frame_scores[i]
                percentage = f"{score:.2f}" if score <= 1.0 else f"{1.0:.2f}"
                
                # Determine assessment based on score
                if score < 0.3:
                    assessment = "Likely Real"
                elif score < 0.5:
                    assessment = "Uncertain"
                elif score < 0.7:
                    assessment = "Suspicious"
                else:
                    assessment = "Likely Fake"
                    
                suspicion_table_data.append([str(i+1), percentage, assessment])
            
            # Create the table
            suspicion_table = Table(suspicion_table_data, colWidths=[1*inch, 1.5*inch, 3.5*inch])
            
            # Style the table
            table_style = [
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ]
            
            # Color code rows based on suspicion level
            for i in range(1, len(suspicion_table_data)):
                score = frame_scores[(i-1)*frame_interval]
                if score >= 0.7:
                    table_style.append(('BACKGROUND', (0, i), (-1, i), colors.pink))
                elif score >= 0.5:
                    table_style.append(('BACKGROUND', (0, i), (-1, i), colors.lightsalmon))
                elif score >= 0.3:
                    table_style.append(('BACKGROUND', (0, i), (-1, i), colors.lightgoldenrodyellow))
            
            suspicion_table.setStyle(TableStyle(table_style))
            story.append(suspicion_table)
            
            # Add a note explaining the table
            story.append(Paragraph(
                "Table shows suspicion scores for selected frames. Pink rows indicate high likelihood of manipulation, "
                "salmon indicates moderate likelihood, and yellow indicates uncertainty.",
                styles['Note']
            ))
            
            story.append(Spacer(1, 0.5*inch))

    # Lip Sync Analysis
    if "lip_sync_analysis" in results_dict:
        story.append(Paragraph("LIP SYNCHRONIZATION ANALYSIS", styles['SectionTitle']))
        lip_sync = results_dict["lip_sync_analysis"]

        # Add explanation of lip sync analysis
        story.append(Paragraph(
            "Lip synchronization analysis examines the correlation between audio speech patterns and visual lip movements. "
            "Deepfakes often show discrepancies in this correlation, as generating perfectly synchronized lip movements "
            "is one of the most challenging aspects of creating convincing fake videos.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.1*inch))

        if isinstance(lip_sync, dict):
            if "real_probability" in lip_sync and "fake_probability" in lip_sync:
                real_prob = lip_sync["real_probability"]
                fake_prob = lip_sync["fake_probability"]

                # Create summary table first
                data = [
                    ["Real Probability", f"{real_prob:.3f} ({real_prob*100:.1f}%)"],
                    ["Fake Probability", f"{fake_prob:.3f} ({fake_prob*100:.1f}%)"]
                ]

                if "description" in lip_sync:
                    data.append(["System Assessment", lip_sync["description"]])

                if "processing_time_seconds" in lip_sync:
                    data.append(["Processing Time", f"{lip_sync['processing_time_seconds']:.2f} seconds"])

                table = Table(data, colWidths=[2*inch, 4*inch])
                table.setStyle(TableStyle([
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                story.append(table)
                
                # Now add the detailed analysis
                story.append(Spacer(1, 0.2*inch))
                story.append(Paragraph("Detailed Analysis", styles['SubSectionTitle']))
                
                # Generate detailed analysis from the analyze_lip_sync function
                detailed_analysis = analyze_lip_sync(lip_sync)
                
                # Split by lines and add as paragraphs
                for para in detailed_analysis.split('\n'):
                    if para.strip():  # Skip empty lines
                        story.append(Paragraph(para, styles['Normal']))
                        story.append(Spacer(1, 0.05*inch))
                
                # Add verdict based on lip sync analysis alone
                story.append(Spacer(1, 0.2*inch))
                if fake_prob > 0.7:
                    story.append(Paragraph("LIP SYNC VERDICT: High probability of manipulation", 
                                         styles['Verdict']))
                elif fake_prob > 0.4:
                    story.append(Paragraph("LIP SYNC VERDICT: Moderate indicators of manipulation", 
                                         styles['Verdict']))
                else:
                    story.append(Paragraph("LIP SYNC VERDICT: Likely authentic lip synchronization", 
                                         styles['VerdictReal']))
                    
            elif isinstance(lip_sync, (int, float, str)):
                story.append(Paragraph(f"Result: {lip_sync}", styles['Normal']))
        
        # Add visual guide to lip sync analysis 
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("How to Interpret Lip Sync Analysis", styles['SubSectionTitle']))
        
        interpretation = [
            "The lip sync analysis provides crucial evidence for detecting deepfakes:",
            "",
            "• <b>High fake probability (>70%):</b> Strong indicator of manipulated content. May show visible misalignment between lip movements and speech.",
            "• <b>Moderate fake probability (40%-70%):</b> Possible manipulation but not conclusive. Requires correlation with other detection methods.",
            "• <b>Low fake probability (<40%):</b> Suggests authentic synchronization between audio and video components.",
            "",
            "Modern deepfakes often struggle with precisely matching lip movements to synthesized speech, especially during complex phonemes or rapid speech patterns."
        ]
        
        for line in interpretation:
            if line:
                story.append(Paragraph(line, styles['Normal']))
            else:
                story.append(Spacer(1, 0.05*inch))
        
        story.append(Spacer(1, 0.3*inch))

    # Grad-CAM and LIME Visualizations
    story.append(Paragraph("Visual Evidence", styles['SubSectionTitle']))
    story.append(Spacer(1, 0.1*inch))

    lime_image = None
    gradcam_image = None
    png_paths = []  # Keep track of PNG paths for cleanup

    if "segmented" in results_dict:
        segmented_data = results_dict["segmented"]
        if isinstance(segmented_data, dict):
            # Check for LIME visualization
            if "LIME" in segmented_data:
                viz_data = segmented_data["LIME"]
                for field in ["overlay", "saliency"]:
                    if isinstance(viz_data, dict) and viz_data.get(field):
                        try:
                            import base64
                            base64_data = viz_data[field]
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

            # Check for GradCAM++ visualization
            if "GradCAM++" in segmented_data:
                viz_data = segmented_data["GradCAM++"]
                for field in ["overlay", "saliency"]:
                    if isinstance(viz_data, dict) and viz_data.get(field):
                        try:
                            import base64
                            base64_data = viz_data[field]
                            if base64_data.startswith('data:image'):
                                base64_data = base64_data.split(',', 1)[1]
                            binary_data = base64.b64decode(base64_data)
                            gradcam_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                            gradcam_file.write(binary_data)
                            gradcam_file.close()
                            gradcam_image = gradcam_file.name
                            print(f"Successfully created temporary GradCAM++ visualization file: {gradcam_image}")
                            break
                        except Exception as e:
                            print(f"Error processing GradCAM++ {field} image: {e}")

    # Add LIME visualization first if available
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

            # Use a unique filename for the PNG
            lime_png_path = lime_image.rsplit('.', 1)[0] + "_lime.png"
            img.save(lime_png_path, "PNG")
            png_paths.append(lime_png_path)
            print(f"Saved LIME visualization as PNG: {lime_png_path}")

            # Add a clear title and the image
            story.append(Paragraph("LIME Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Areas highlighted by LIME algorithm showing regions that influenced the detection decision", styles['Note']))
            img_obj = RLImage(lime_png_path, width=display_width, height=display_height)
            story.append(img_obj)

            # Add significant space after this visualization
            story.append(Spacer(1, 0.5*inch))
        except Exception as e:
            story.append(Paragraph(f"Error loading LIME visualization image: {str(e)}", styles['Note']))
            print(f"Exception when adding LIME visualization to PDF: {e}")

    # Add GradCAM++ visualization if available
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

            # Use a unique filename for the PNG
            gradcam_png_path = gradcam_image.rsplit('.', 1)[0] + "_gradcam.png"
            img.save(gradcam_png_path, "PNG")
            png_paths.append(gradcam_png_path)
            print(f"Saved GradCAM++ visualization as PNG: {gradcam_png_path}")

            # Consider adding a page break if the images are large
            if display_height > 3 * inch:
                story.append(PageBreak())

            # Add a clear title and the image
            story.append(Paragraph("GradCAM++ Visualization", styles['SubSectionTitle']))
            story.append(Paragraph("Areas highlighted by GradCAM++ algorithm showing regions of potential manipulation", styles['Note']))
            img_obj = RLImage(gradcam_png_path, width=display_width, height=display_height)
            story.append(img_obj)
        except Exception as e:
            story.append(Paragraph(f"Error loading GradCAM++ visualization image: {str(e)}", styles['Note']))
            print(f"Exception when adding GradCAM++ visualization to PDF: {e}")

    # Continue with PDF generation...
    doc.build(story, onFirstPage=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name),
             onLaterPages=lambda canvas, doc: create_header_footer(canvas, doc, case_number, investigator_name))

    # Clean up temporary files after PDF generation
    for tmp in [lime_image, gradcam_image] + png_paths:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
                print(f"Cleaned up temporary file: {tmp}")
            except Exception as e:
                print(f"Error cleaning up temporary file {tmp}: {e}")

    return output_path
def create_header_footer(canvas, doc, case_number, investigator_name):
    """Create header and footer for each page"""
    canvas.saveState()
    
    # Header
    canvas.setFont('Helvetica-Bold', 10)
    canvas.drawString(inch, 10.5 * inch, f"CASE #{case_number}")
    canvas.setFont('Helvetica', 9)
    canvas.drawString(inch, 10.3 * inch, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    canvas.drawString(inch, 10.1 * inch, f"Investigator: {investigator_name}")
    
    # Add a line under the header
    canvas.line(inch, 10*inch, 7.5*inch, 10*inch)
    
    # Footer
    canvas.setFont('Helvetica', 8)
    canvas.line(inch, 0.75*inch, 7.5*inch, 0.75*inch)
    canvas.drawString(inch, 0.5 * inch, "CONFIDENTIAL - FORENSIC INVESTIGATION REPORT")
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")
    
    canvas.restoreState()

def analyze_lip_sync(lip_sync_data):
    """
    Generate detailed analysis of lip sync results
    
    Args:
        lip_sync_data (dict): Lip sync analysis results
        
    Returns:
        str: Detailed analysis text
    """
    if not lip_sync_data or not isinstance(lip_sync_data, dict):
        return "Insufficient data for lip sync analysis."
        
    real_prob = lip_sync_data.get('real_probability', None)
    fake_prob = lip_sync_data.get('fake_probability', None)
    description = lip_sync_data.get('description', '')
    
    if real_prob is None or fake_prob is None:
        return "Incomplete lip sync data available."
    
    analysis = []
    analysis.append("<b>Detailed Lip Synchronization Analysis</b>")
    analysis.append("")
    
    # Confidence level description
    confidence = "high" if max(real_prob, fake_prob) > 0.8 else "moderate" if max(real_prob, fake_prob) > 0.6 else "low"
    
    if fake_prob > real_prob:
        analysis.append(f"The analysis detected lip sync inconsistencies with {confidence} confidence ({fake_prob*100:.1f}%). "
                       f"This suggests possible manipulation in how the audio aligns with the visual mouth movements.")
        
        if fake_prob > 0.8:
            analysis.append("The high probability of fake lip synchronization strongly indicates this video has been manipulated. "
                           "Common methods include deepfake face swaps, dubbed audio, or AI-generated speech synthesis.")
        elif fake_prob > 0.6:
            analysis.append("The moderate probability of fake lip synchronization suggests potential manipulation, though not definitive. "
                          "Further examination of specific frames where audio and visual elements appear misaligned is recommended.")
        else:
            analysis.append("While some lip sync anomalies were detected, the confidence level is relatively low. "
                          "This could indicate subtle manipulation or simply natural variation in speech patterns.")
    else:
        analysis.append(f"The lip synchronization appears natural with {confidence} confidence ({real_prob*100:.1f}%). "
                       f"The speech patterns and mouth movements exhibit consistent timing and natural correlation.")
        
        if real_prob > 0.8:
            analysis.append("The high probability of authentic lip synchronization strongly supports the video's authenticity. "
                          "The natural alignment between speech audio and facial movements is difficult to fake convincingly.")
    
    # Technical explanation
    analysis.append("")
    analysis.append("<b>Technical Methodology:</b> The lip sync detection algorithm analyzes the temporal correlation between "
                  "audio phonemes (speech sounds) and visual mouth shape/movements. Deepfakes often show microsecond-level "
                  "discrepancies in this correlation, as generating perfectly synchronized lip movements is challenging even "
                  "for sophisticated AI systems.")
    
    # Add the original description if available
    if description:
        analysis.append("")
        analysis.append(f"<b>System Assessment:</b> {description}")
    
    return "\n".join(analysis)

def generate_video_insights(video_analysis):
    """
    Generate AI-driven insights about the video analysis
    
    Args:
        video_analysis (dict): Video analysis data
        
    Returns:
        dict: Dictionary of insights for different sections
    """
    insights = {}
    
    # Generate executive summary
    if isinstance(video_analysis, dict):
        # Extract key metrics
        is_fake = video_analysis.get("overall_assessment", {}).get("is_likely_fake", False)
        fake_percentage = video_analysis.get("overall_assessment", {}).get("fake_frame_percentage", 0)
        lip_sync_result = video_analysis.get("lip_sync_analysis", {})
        frame_count = video_analysis.get("video_analysis", {}).get("total_frames_analyzed", 0)
        fake_frames = video_analysis.get("video_analysis", {}).get("fake_frames_detected", 0)
        
        # Create executive summary
        summary = []
        if is_fake:
            summary.append(f"This video analysis indicates a <b>high probability of manipulation</b> with {fake_percentage*100:.1f}% of analyzed frames showing signs of deepfake artifacts. "
                          f"Out of {frame_count} frames analyzed, {fake_frames} frames showed significant indicators of manipulation.")
        else:
            summary.append(f"This video analysis indicates a <b>low probability of manipulation</b> with only {fake_percentage*100:.1f}% of analyzed frames showing potential deepfake artifacts. "
                          f"Out of {frame_count} frames analyzed, only {fake_frames} frames showed potential indicators of manipulation.")
        
        # Add lip sync info to summary
        if isinstance(lip_sync_result, dict) and "fake_probability" in lip_sync_result:
            lip_fake_prob = lip_sync_result.get("fake_probability", 0)
            if lip_fake_prob > 0.7:
                summary.append(f"The lip synchronization analysis strongly supports this finding, showing {lip_fake_prob*100:.1f}% probability of manipulation between audio and visual elements. "
                              f"This suggests sophisticated methods were used to fabricate or alter speech content in this video.")
            elif lip_fake_prob > 0.4:
                summary.append(f"The lip synchronization analysis shows moderate indicators of manipulation with {lip_fake_prob*100:.1f}% probability of altered speech-facial coordination. "
                              f"This suggests possible audio-visual inconsistencies that should be examined more closely.")
            else:
                summary.append(f"Despite other findings, the lip synchronization analysis shows relatively natural speech-facial coordination with only {lip_fake_prob*100:.1f}% probability of manipulation. "
                              f"This suggests any potential manipulation may have preserved the original audio or used advanced neural rendering techniques.")
        
        insights["executive_summary"] = "\n".join(summary)
        
        # Generate technical analysis
        tech_analysis = []
        
        # Frame analysis insights
        frame_results = video_analysis.get("video_analysis", {}).get("results", [])
        if frame_results:
            # Find patterns in frame scores
            frame_scores = [f.get("fusion_score", 0) for f in frame_results if "fusion_score" in f]
            if frame_scores:
                avg_score = sum(frame_scores) / len(frame_scores)
                max_score = max(frame_scores)
                min_score = min(frame_scores)
                variance = sum((x - avg_score) ** 2 for x in frame_scores) / len(frame_scores)
                
                tech_analysis.append(f"<b>Frame Analysis Metrics:</b> Average manipulation score: {avg_score:.3f}, Maximum: {max_score:.3f}, Minimum: {min_score:.3f}, Variance: {variance:.3f}")
                
                if variance > 0.1:
                    tech_analysis.append("The high variance in frame scores indicates inconsistent manipulation across the video. This pattern is common in videos where only specific segments have been altered.")
                else:
                    tech_analysis.append("The consistent pattern of detection scores across frames suggests either uniform quality throughout the video or consistent application of manipulation techniques.")
        
        # Add technical findings
        if video_analysis.get("frame_scores"):
            consecutive_peaks = 0
            peak_count = 0
            for score in video_analysis.get("frame_scores"):
                if score > 0.7:
                    consecutive_peaks += 1
                    peak_count += 1
                else:
                    consecutive_peaks = 0
                    
                if consecutive_peaks >= 3:
                    tech_analysis.append(f"Detected sustained high manipulation probability across multiple consecutive frames, suggesting deliberate and focused content manipulation rather than compression artifacts.")
                    break
                    
            if peak_count > 0:
                peak_percentage = peak_count / len(video_analysis.get("frame_scores"))
                tech_analysis.append(f"{peak_percentage*100:.1f}% of frames showed high manipulation probability (>0.7), concentrated primarily in specific video segments.")
        
        insights["technical_analysis"] = "\n".join(tech_analysis)
        
        # Generate recommendations
        recommendations = []
        if is_fake:
            recommendations.append("<b>Primary Recommendation:</b> This video should be treated as potentially manipulated content. Consider the following actions:")
            recommendations.append("• Request the original source video file with metadata intact for further forensic analysis")
            recommendations.append("• Examine the context in which this video was shared or distributed")
            recommendations.append("• If this video relates to a legal matter, consult with digital forensics specialists")
            recommendations.append("• Confirm key events depicted in the video through independent sources")
        else:
            recommendations.append("<b>Primary Recommendation:</b> While this analysis indicates a low probability of manipulation, consider the following precautions:")
            recommendations.append("• Verify the chain of custody for this video file")
            recommendations.append("• Check file metadata for signs of editing or processing")
            recommendations.append("• For critical applications, consider additional verification through alternate detection methods")
        
        insights["recommendations"] = "\n".join(recommendations)
    
    return insights