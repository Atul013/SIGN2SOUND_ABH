
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def create_report():
    doc = SimpleDocTemplate("results/Technical_Report.pdf", pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    Story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles["Heading1"]
    title_style.alignment = 1 # Center
    
    subtitle_style = styles["Heading2"]
    subtitle_style.textColor = colors.darkblue
    
    normal_style = styles["Normal"]
    normal_style.spaceAfter = 12
    
    # --- Title Page ---
    Story.append(Paragraph("Sign2Sound: Technical Performance Report", title_style))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("<b>Project:</b> Sign2Sound - Real-time ASL to Speech System", normal_style))
    Story.append(Paragraph("<b>Date:</b> January 29, 2026", normal_style))
    Story.append(Spacer(1, 24))
    
    # --- Section 1: Quantitative Performance ---
    Story.append(Paragraph("1. Quantitative Performance Metrics", subtitle_style))
    Story.append(Paragraph("The model was evaluated on a held-out validation set of <b>27,902</b> samples.", normal_style))
    
    data = [
        ['Metric', 'Score'],
        ['Accuracy', '97.63%'],
        ['Precision', '97.66%'],
        ['Recall', '97.63%'],
        ['F1-Score', '97.63%']
    ]
    t = Table(data, colWidths=[200, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    Story.append(t)
    Story.append(Spacer(1, 24))
    
    # --- Section 2: Training Dynamics ---
    Story.append(Paragraph("2. Training Dynamics", subtitle_style))
    Story.append(Paragraph("The model showed consistent convergence over 10 epochs. Validation accuracy improved from 93.2% to 97.6%.", normal_style))
    
    graph_path = "results/training_curves.png"
    if os.path.exists(graph_path):
        im = Image(graph_path, width=6*inch, height=3*inch)
        Story.append(im)
        Story.append(Spacer(1, 12))
    
    # --- Section 3: Per-Class Performance ---
    Story.append(Paragraph("3. Per-Class Performance Analysis", subtitle_style))
    Story.append(Paragraph("<b>Best Performing Classes (>99% F1):</b>", normal_style))
    Story.append(Paragraph("• F, B, W, Space, L", normal_style))
    
    Story.append(Paragraph("<b>Challenging Classes:</b>", normal_style))
    Story.append(Paragraph("• <b>N & M:</b> High confusion due to similarity (2 fingers vs 3 fingers over thumb).", normal_style))
    Story.append(Paragraph("• <b>S:</b> Confused with 'A' and 'E' (similar fist shapes).", normal_style))
    
    # --- Section 4: Confusion Matrix ---
    Story.append(Paragraph("4. Confusion Matrix Visualization", subtitle_style))
    cm_path = "results/confusion_matrix.png"
    if os.path.exists(cm_path):
        im = Image(cm_path, width=6*inch, height=4*inch)
        Story.append(im)
    
    Story.append(Spacer(1, 24))
    
    # --- Section 5: Inference Speed ---
    Story.append(Paragraph("5. Inference Speed & Real-time Capability", subtitle_style))
    speed_data = [
        ['Component', 'Latency'],
        ['MediaPipe (Feature Extraction)', '~35 ms'],
        ['Classifier Inference', '< 1 ms'],
        ['Total Pipeline', '~40 ms (25 FPS)']
    ]
    t2 = Table(speed_data, colWidths=[200, 150])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    Story.append(t2)
    
    # Build Document
    doc.build(Story)
    print("PDF Report generated successfully!")

if __name__ == "__main__":
    create_report()
