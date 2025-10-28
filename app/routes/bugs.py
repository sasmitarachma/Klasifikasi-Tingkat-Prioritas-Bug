from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, send_file, after_this_request, jsonify
import os
import pandas as pd
import io
import csv
from datetime import datetime, timedelta
import tempfile
from textwrap import shorten
from io import BytesIO
import base64
from collections import Counter

# Import ML models
from app.utils.model.dl_model import BugPriorityClassifier
from app.utils.model.github_scanner import GithubScanner

bugs = Blueprint('bugs', __name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

# Simple Bug Data class (tanpa database model)
class BugResult:
    def __init__(self, summary, preprocessed_summary, predicted_priority, confidence_score):
        self.summary = summary
        self.preprocessed_summary = preprocessed_summary
        self.predicted_priority = predicted_priority
        self.confidence_score = confidence_score

@bugs.route('/', methods=['GET', 'POST'])
def index():
    result = []
    classifier = BugPriorityClassifier()

    if request.method == 'POST':
        summary = request.form.get('summary', '').strip()
        file = request.files.get('file')
        repo_url = request.form.get('repo_url', '').strip()

        if summary:
            tokens = classifier.preprocess_text(summary)
            preprocessed_text = ' '.join(tokens)
            priority, confidence = classifier.predict_priority([preprocessed_text])
            bug_result = BugResult(
                summary=summary,
                preprocessed_summary=preprocessed_text,
                predicted_priority=priority[0],
                confidence_score=confidence[0]
            )
            result.append(bug_result)

        elif file and allowed_file(file.filename):
            try:
                df = pd.read_csv(file)
                for _, row in df.iterrows():
                    s = str(row.get('Summary', '')).strip()
                    if not s:
                        continue
                    tokens = classifier.preprocess_text(s)
                    preprocessed_text = ' '.join(tokens)
                    priority, confidence = classifier.predict_priority([preprocessed_text])

                    bug_result = BugResult(
                        summary=s,
                        preprocessed_summary=preprocessed_text,
                        predicted_priority=priority[0],
                        confidence_score=confidence[0]
                    )
                    result.append(bug_result)
            except Exception as e:
                flash(f"Failed to process CSV file: {str(e)}", "danger")
                return redirect(url_for("bugs.index"))

        elif repo_url:
            if "github.com" not in repo_url:
                flash("Please enter a valid GitHub repository URL.", "danger")
                return redirect(url_for("bugs.index"))

            try:
                scanner = GithubScanner(repo_url)
                summaries = scanner.scan_repo()

                for s in summaries:
                    tokens = classifier.preprocess_text(s)
                    preprocessed_text = ' '.join(tokens)
                    priority, confidence = classifier.predict_priority([preprocessed_text])

                    bug_result = BugResult(
                        summary=s,
                        preprocessed_summary=preprocessed_text,
                        predicted_priority=priority[0],
                        confidence_score=confidence[0]
                    )
                    result.append(bug_result)
            except Exception as e:
                flash(f"Failed to scan GitHub repository: {str(e)}", "danger")
                return redirect(url_for("bugs.index"))

        else:
            flash('Please provide a summary, upload a file, or enter a GitHub repository URL.', 'danger')
            return redirect(url_for('bugs.index'))

    # Hitung priority_counts dari result
    counts = Counter([bug.predicted_priority for bug in result])
    priority_counts = {
        'Tinggi': counts.get('Tinggi', 0),
        'Sedang': counts.get('Sedang', 0),
        'Rendah': counts.get('Rendah', 0)
    }
    total_bugs = len(result)

    return render_template('bugs/input.html',
                           result=result,
                           priority_counts=priority_counts,
                           total_bugs=total_bugs)

# Route untuk export hasil (tambahan)
@bugs.route('/export_results/<format>', methods=['POST'])
def export_results(format):
    """Export hasil analisis saat ini ke CSV atau PDF"""
    if format not in ['csv', 'pdf']:
        flash('Invalid format', 'danger')
        return redirect(url_for('bugs.index'))
    
    # Ambil data dari form POST (dari JavaScript)
    results_data = request.get_json()
    
    if not results_data:
        flash('No data to export', 'danger')
        return redirect(url_for('bugs.index'))

    if format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['No', 'Summary', 'Preprocessed Summary', 'Priority', 'Confidence Score'])

        for i, bug in enumerate(results_data, 1):
            writer.writerow([
                i,
                bug.get('summary', ''),
                bug.get('preprocessed_summary', ''),
                bug.get('predicted_priority', ''),
                f"{bug.get('confidence_score', 0):.3f}"
            ])

        output.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'bug_analysis_results_{timestamp}.csv'
        )

    elif format == 'pdf':
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
            from reportlab.lib import colors
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            doc = SimpleDocTemplate(temp_file.name, pagesize=A4)

            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = styles['Title']
            elements.append(Paragraph('Bug Priority Classification Results', title_style))
            elements.append(Spacer(1, 12))
            

            # Table
            data = [['No', 'Summary', 'Priority', 'Confidence']]
            for i, bug in enumerate(results_data, 1):
                data.append([
                    str(i),
                    shorten(bug.get('summary', ''), width=60, placeholder="..."),
                    bug.get('predicted_priority', ''),
                    f"{bug.get('confidence_score', 0):.3f}"
                ])

            table = Table(data, colWidths=[0.5*inch, 4*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))

            elements.append(table)
            doc.build(elements)

            @after_this_request
            def remove_temp_file(response):
                try:
                    os.remove(temp_file.name)
                except Exception as e:
                    current_app.logger.error(f'Error deleting temp file: {e}')
                return response

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return send_file(
                temp_file.name,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'bug_analysis_results_{timestamp}.pdf'
            )
        except ImportError:
            flash('PDF generation requires the reportlab library. Please install it.', 'danger')
            return jsonify({'error': 'PDF library not available'}), 500
        except Exception as e:
            flash(f'Error generating PDF: {str(e)}', 'danger')
            return jsonify({'error': str(e)}), 500