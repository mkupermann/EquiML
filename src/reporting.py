from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import os

def generate_html_report(metrics, output_path='evaluation_report.html', template_path='src/report_template.html'):
    """
    Generates a comprehensive HTML report from evaluation metrics.

    Args:
        metrics (dict): A dictionary containing the evaluation metrics.
        output_path (str): The path to save the HTML report.
        template_path (str): The path to the Jinja2 template file.
    """
    template_dir = os.path.dirname(template_path)
    template_name = os.path.basename(template_path)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)

    html_content = template.render(
        metrics=metrics,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    with open(output_path, 'w') as f:
        f.write(html_content)
